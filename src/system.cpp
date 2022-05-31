#include "rcvio/system.hpp"

#include <fstream>
#include <iostream>

#include <boost/thread.hpp>

#include <ros/package.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include "rcvio/numerics.h"

namespace rcvio
{
    nav_msgs::Path path;

    std::ofstream pose_results;
    std::ofstream time_results;

    System::System(const std::string &settings_file)
    {
        std::cout << "\n"
                  << "RC-VIO: Robot Centric Visual-Inertial Odometry"
                  << "\n"
                  << "\n";

        cv::FileStorage fs_settings(settings_file, cv::FileStorage::READ);
        if (!fs_settings.isOpened())
        {
            ROS_ERROR("Failed to open settings file at: %s", settings_file.c_str());
            exit(-1);
        }

        imu_rate_ = fs_settings["IMU.dps"];

        sigma_gyro_noise_ = fs_settings["IMU.sigma_g"];
        sigma_gyro_bias_ = fs_settings["IMU.sigma_wg"];
        sigma_accel_noise_ = fs_settings["IMU.sigma_a"];
        sigma_accel_bias_ = fs_settings["IMU.sigma_wa"];

        gravity_ = fs_settings["IMU.nG"];

        cam_time_offset_ = fs_settings["Camera.nTimeOffset"];

        const int max_tracking_length = fs_settings["Tracker.nMaxTrackingLength"];
        sliding_window_size_ = max_tracking_length - 1;

        const int min_tracking_length = fs_settings["Tracker.nMinTrackingLength"];
        min_clone_states_ = min_tracking_length - 1;

        const int enable_alignment = fs_settings["INI.EnableAlignment"];
        enable_alignment_ = enable_alignment;

        const int record_outputs = fs_settings["INI.RecordOutputs"];
        record_outputs_ = record_outputs;

        if (record_outputs_)
        {
            std::string pkg_path = ros::package::getPath("rcvio");
            pose_results.open(pkg_path + "/stamped_pose_ests.dat", std::ofstream::out);
            time_results.open(pkg_path + "/time_cost.dat", std::ofstream::out);
        }

        threshold_angle_ = fs_settings["INI.nThresholdAngle"];
        threshold_displacement_ = fs_settings["INI.nThresholdDisplacement"];

        is_moving_ = false;
        is_ready_ = false;

        input_buffer_.reset(new InputBuffer());
        tracker_.reset(new Tracker(fs_settings));
        updater_.reset(new Updater(fs_settings));
        pre_integrator_.reset(new PreIntegrator(fs_settings));

        path_publish_ = system_node_.advertise<nav_msgs::Path>("/rcvio/trajectory", 1);
        odometry_publish_ = system_node_.advertise<nav_msgs::Odometry>("/rcvio/odometry", 1);
    }

    void System::initialize(const Eigen::Vector3d &w, const Eigen::Vector3d &a,
                            const int imu_data, const bool enable_alignment)
    {
        Eigen::Vector3d g = a;
        g.normalize();

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (enable_alignment)
        {
            Eigen::Vector3d zv = g;

            Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
            Eigen::Vector3d xv = ex - zv * zv.transpose() * ex;
            xv.normalize();

            Eigen::Vector3d yv = skew(zv) * xv;
            yv.normalize();

            Eigen::Matrix3d temp_R;
            temp_R << xv, yv, zv;
            R = temp_R;
        }

        xkk.setZero(26, 1);
        xkk.block(0, 0, 4, 1) = rotToQuat(R);
        xkk.block(7, 0, 3, 1) = g;

        if (imu_data > 1)
        {
            xkk.block(20, 0, 3, 1) = w;
            xkk.block(23, 0, 3, 1) = a - gravity_ * g;
        }

        double dt = 1.0 / imu_rate_;

        Pkk.setZero(24, 24);
        Pkk(0, 0) = std::pow(1e-3, 2);
        Pkk(1, 1) = std::pow(1e-3, 2);
        Pkk(2, 2) = std::pow(1e-3, 2);
        Pkk(3, 3) = std::pow(1e-3, 2);
        Pkk(4, 4) = std::pow(1e-3, 2);
        Pkk(5, 5) = std::pow(1e-3, 2);
        Pkk(6, 6) = imu_data * dt * std::pow(sigma_accel_noise_, 2);
        Pkk(7, 7) = imu_data * dt * std::pow(sigma_accel_noise_, 2);
        Pkk(8, 8) = imu_data * dt * std::pow(sigma_accel_noise_, 2);
        Pkk(18, 18) = imu_data * dt * std::pow(sigma_gyro_bias_, 2);
        Pkk(19, 19) = imu_data * dt * std::pow(sigma_gyro_bias_, 2);
        Pkk(20, 20) = imu_data * dt * std::pow(sigma_gyro_bias_, 2);
        Pkk(21, 21) = imu_data * dt * std::pow(sigma_accel_bias_, 2);
        Pkk(22, 22) = imu_data * dt * std::pow(sigma_accel_bias_, 2);
        Pkk(23, 23) = imu_data * dt * std::pow(sigma_accel_bias_, 2);
    }

    void System::monoVIO()
    {
        static int clone_states = 0;
        static int image_count_after_init = 0;

        std::pair<ImageData *, std::list<ImuData *>> measurements;
        if (!input_buffer_->getMeasurements(cam_time_offset_, measurements))
        {
            return;
        }

        if (!is_ready_)
        {
            static Eigen::Vector3d wm = Eigen::Vector3d::Zero();
            static Eigen::Vector3d am = Eigen::Vector3d::Zero();
            static int imu_data_count = 0;

            if (!is_moving_)
            {
                Eigen::Vector3d angle;
                Eigen::Vector3d velocity;
                Eigen::Vector3d displacement;
                angle.setZero();
                velocity.setZero();
                displacement.setZero();

                for (std::list<ImuData *>::const_iterator it = measurements.second.begin();
                     it != measurements.second.end(); ++it)
                {
                    Eigen::Vector3d w = (*it)->wm_;
                    Eigen::Vector3d a = (*it)->am_;
                    double dt = (*it)->dt_;

                    a -= gravity_ * a / a.norm();

                    angle += dt * w;
                    velocity += dt * a;
                    displacement += dt * velocity + 0.5 * std::pow(dt, 2) * a;
                }

                if (angle.norm() > threshold_angle_ || displacement.norm() > threshold_displacement_)
                {
                    is_moving_ = true;
                }
            }

            while (!measurements.second.empty())
            {
                if (!is_moving_)
                {
                    wm += (measurements.second.front())->wm_;
                    am += (measurements.second.front())->am_;
                    measurements.second.pop_front();
                    imu_data_count++;
                }
                else
                {
                    if (imu_data_count == 0)
                    {
                        wm = (measurements.second.front())->wm_;
                        am = (measurements.second.front())->am_;
                        imu_data_count = 1;
                    }
                    else
                    {
                        wm = wm / imu_data_count;
                        am = am / imu_data_count;
                    }

                    initialize(wm, am, imu_data_count, enable_alignment_);

                    is_ready_ = true;
                    break;
                }
            }

            if (!is_ready_)
            {
                return;
            }
        }

        image_count_after_init++;

        ros::WallTime t1, t2, t3;

        t1 = ros::WallTime::now();

        tracker_->track(measurements.first->image_, measurements.second);

        t2 = ros::WallTime::now();

        pre_integrator_->propagate(xkk, Pkk, measurements.second);

        if (clone_states > min_clone_states_)
        {
            updater_->update(pre_integrator_->xk1k,
                             pre_integrator_->Pk1k,
                             tracker_->feature_types_for_update_,
                             tracker_->feature_measurements_for_update_);

            xkk = updater_->xk1k1;
            Pkk = updater_->Pk1k1;
        }
        else
        {
            xkk = pre_integrator_->xk1k;
            Pkk = pre_integrator_->Pk1k;
        }

        if (image_count_after_init > 1)
        {
            if (clone_states < sliding_window_size_)
            {
                Eigen::MatrixXd tempx(26 + 7 * (clone_states + 1), 1);
                tempx << xkk, xkk.block(10, 0, 7, 1);
                xkk = tempx;

                Eigen::MatrixXd J(24 + 6 * (clone_states + 1), 24 + 6 * clone_states);
                J.setZero();
                J.block(0, 0, 24 + 6 * clone_states, 24 + 6 * clone_states).setIdentity();
                J.block(24 + 6 * clone_states, 9, 3, 3).setIdentity();
                J.block(24 + 6 * clone_states + 3, 12, 3, 3).setIdentity();

                Eigen::MatrixXd temp_P = J * Pkk * (J.transpose());
                temp_P = 0.5 * (temp_P + temp_P.transpose());
                Pkk = temp_P;

                clone_states++;
            }
            else
            {
                Eigen::MatrixXd tempx(26 + 7 * sliding_window_size_, 1);
                tempx << xkk.block(0, 0, 26, 1), xkk.block(26 + 7, 0, 7 * (sliding_window_size_ - 1), 1), xkk.block(10, 0, 7, 1);
                xkk = tempx;

                Eigen::MatrixXd J(24 + 6 * (sliding_window_size_ + 1), 24 + 6 * sliding_window_size_);
                J.setZero();
                J.block(0, 0, 24 + 6 * sliding_window_size_, 24 + 6 * sliding_window_size_).setIdentity();
                J.block(24 + 6 * sliding_window_size_, 9, 3, 3).setIdentity();
                J.block(24 + 6 * sliding_window_size_ + 3, 12, 3, 3).setIdentity();

                Eigen::MatrixXd temp_P = J * Pkk * (J.transpose());
                temp_P = 0.5 * (temp_P + temp_P.transpose());
                Pkk.block(0, 0, 24, 24) = temp_P.block(0, 0, 24, 24);
                Pkk.block(0, 24, 24, 6 * sliding_window_size_) = temp_P.block(0, 24 + 6, 24, 6 * sliding_window_size_);
                Pkk.block(24, 0, 6 * sliding_window_size_, 24) = temp_P.block(24 + 6, 0, 6 * sliding_window_size_, 24);
                Pkk.block(24, 24, 6 * sliding_window_size_, 6 * sliding_window_size_) = temp_P.block(24 + 6, 24 + 6, 6 * sliding_window_size_, 6 * sliding_window_size_);
            }
        }

        Eigen::Vector4d qG = xkk.block(0, 0, 4, 1);
        Eigen::Vector3d pG = xkk.block(4, 0, 3, 1);
        Eigen::Vector3d gk = xkk.block(7, 0, 3, 1);
        Eigen::Vector4d qk = xkk.block(10, 0, 4, 1);
        Eigen::Vector3d pk = xkk.block(14, 0, 3, 1);
        Eigen::Vector3d vk = xkk.block(17, 0, 3, 1);

        Eigen::Matrix3d RG = quatToRot(qG);
        Eigen::Matrix3d Rk = quatToRot(qk);

        gk = Rk * gk;
        gk.normalize();

        Eigen::Vector4d qkG = quatMul(qk, qG);
        Eigen::Vector3d pkG = Rk * (pG - pk);
        Eigen::Vector3d pGk = RG.transpose() * (pk - pG);

        Eigen::Matrix<double, 24, 24> Vk;
        Vk.setZero();
        Vk.block(0, 0, 3, 3) = Rk;
        Vk.block(0, 9, 3, 3).setIdentity();
        Vk.block(3, 3, 3, 3) = Rk;
        Vk.block(3, 9, 3, 3) = skew(pkG);
        Vk.block(3, 12, 3, 3) = -Rk;
        Vk.block(6, 6, 3, 3) = Rk;
        Vk.block(6, 9, 3, 3) = skew(gk);
        Vk.block(15, 15, 9, 9).setIdentity();

        Pkk.block(0, 0, 24, 24) = Vk * Pkk.block(0, 0, 24, 24) * (Vk.transpose());
        Pkk.block(0, 24, 24, 6 * clone_states) = Vk * Pkk.block(0, 24, 24, 6 * clone_states);
        Pkk.block(24, 0, 6 * clone_states, 24) = Pkk.block(0, 24, 24, 6 * clone_states).transpose();
        Pkk = .5 * (Pkk + Pkk.transpose());

        xkk.block(0, 0, 4, 1) = qkG;
        xkk.block(4, 0, 3, 1) = pkG;
        xkk.block(7, 0, 3, 1) = gk;
        xkk.block(10, 0, 4, 1) = Eigen::Vector4d(0, 0, 0, 1);
        xkk.block(14, 0, 3, 1) = Eigen::Vector3d(0, 0, 0);

        t3 = ros::WallTime::now();

        if (record_outputs_)
        {
            pose_results << std::setprecision(19) << measurements.first->ts_ << " "
                         << pGk(0) << " " << pGk(1) << " " << pGk(2) << " "
                         << qkG(0) << " " << qkG(1) << " " << qkG(2) << " " << qkG(3) << "\n";
            pose_results.flush();

            time_results << image_count_after_init << std::setprecision(19) << " "
                         << 1e3 * (t2.toSec() - t1.toSec()) << " "
                         << 1e3 * (t3.toSec() - t2.toSec()) << "\n";
            time_results.flush();
        }

        ROS_INFO("qkG: %5f, %5f, %5f, %5f", qkG(0), qkG(1), qkG(2), qkG(3));
        ROS_INFO("pGk: %5f, %5f, %5f\n", pGk(0), pGk(1), pGk(2));

        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = ros::Time::now();
        transform_stamped.header.frame_id = "world";
        transform_stamped.child_frame_id = "imu";
        transform_stamped.transform.translation.x = pGk(0);
        transform_stamped.transform.translation.y = pGk(1);
        transform_stamped.transform.translation.z = pGk(2);
        transform_stamped.transform.rotation.x = qkG(0);
        transform_stamped.transform.rotation.y = qkG(1);
        transform_stamped.transform.rotation.z = qkG(2);
        transform_stamped.transform.rotation.w = qkG(3);

        tf_publish_.sendTransform(transform_stamped);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time::now();
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = pGk(0);
        odometry.pose.pose.position.y = pGk(1);
        odometry.pose.pose.position.z = pGk(2);
        odometry.pose.pose.orientation.x = qkG(0);
        odometry.pose.pose.orientation.y = qkG(1);
        odometry.pose.pose.orientation.z = qkG(2);
        odometry.pose.pose.orientation.w = qkG(3);
        odometry.child_frame_id = "imu";
        odometry.twist.twist.linear.x = vk(0);
        odometry.twist.twist.linear.y = vk(1);
        odometry.twist.twist.linear.z = vk(2);

        odometry_publish_.publish(odometry);

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "world";
        pose.pose.position.x = pGk(0);
        pose.pose.position.y = pGk(1);
        pose.pose.position.z = pGk(2);
        pose.pose.orientation.x = qkG(0);
        pose.pose.orientation.y = qkG(1);
        pose.pose.orientation.z = qkG(2);
        pose.pose.orientation.w = qkG(3);

        path.header.frame_id = "world";
        path.poses.push_back(pose);

        path_publish_.publish(path);

        usleep(1000);
    }
}