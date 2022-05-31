#include "rcvio/updater.hpp"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <visualization_msgs/Marker.h>

#include "rcvio/numerics.h"

namespace rcvio
{
    static int cloud_id = 0;
    std_msgs::ColorRGBA color_landmark;
    geometry_msgs::Vector3 scale_landmark;

    Updater::Updater(const cv::FileStorage &fs_settings)
    {
        cam_rate_ = fs_settings["Camera.fps"];

        sigma_image_noise_x_ = fs_settings["Camera.sigma_px"];
        sigma_image_noise_y_ = fs_settings["Camera.sigma_py"];

        cv::Mat T(4, 4, CV_32F);
        fs_settings["Camera.T_BC0"] >> T;
        Eigen::Matrix4d Tic;
        cv::cv2eigen(T, Tic);
        Ric_ = Tic.block<3, 3>(0, 0);
        tic_ = Tic.block<3, 1>(0, 3);
        Rci_ = Ric_.transpose();
        tci_ = -Rci_ * tic_;

        xk1k1.setZero(26, 1);
        Pk1k1.setZero(24, 24);

        feature_publish_ = updater_node_.advertise<visualization_msgs::Marker>("/rcvio/landmarks", 1);
        publish_rate_ = fs_settings["Landmark.nPubRate"];

        scale_landmark.x = fs_settings["Landmark.nScale"];
        scale_landmark.y = fs_settings["Landmark.nScale"];
        scale_landmark.z = fs_settings["Landmark.nScale"];

        color_landmark.a = 1;
        color_landmark.r = 0;
        color_landmark.b = 1;
        color_landmark.g = 0;
    }

    void Updater::update(Eigen::VectorXd &xk1k,
                         Eigen::MatrixXd &Pk1k,
                         std::vector<unsigned char> &feature_types_for_update,
                         std::vector<std::list<cv::Point2f>> &feature_measurements_for_update)
    {
        visualization_msgs::Marker cloud;
        cloud.header.frame_id = "imu";
        cloud.ns = "points";
        cloud.id = ++cloud_id;
        cloud.color = color_landmark;
        cloud.scale = scale_landmark;
        cloud.pose.orientation.w = 1.0;
        cloud.lifetime = ros::Duration(1 / publish_rate_);
        cloud.action = visualization_msgs::Marker::ADD;
        cloud.type = visualization_msgs::Marker::POINTS;

        int num_feature = static_cast<int>(feature_types_for_update.size());

        int rows = 0;
        for (int i = 0; i < num_feature; ++i)
        {
            rows += 2 * static_cast<int>(feature_measurements_for_update.at(i).size());
        }

        int clone_states = (xk1k.rows() - 26) / 7;

        Eigen::VectorXd r(rows, 1);
        Eigen::MatrixXd Hx(rows, 24 + 6 * clone_states);
        r.setZero();
        Hx.setZero();

        int row_count = 0;
        int good_feature_count = 0;

        for (int feature_idx = 0; feature_idx < num_feature; ++feature_idx)
        {
            char feature_type = feature_types_for_update.at(feature_idx);
            std::list<cv::Point2f> feature_measurements = feature_measurements_for_update.at(feature_idx);

            int track_length = static_cast<int>(feature_measurements.size());
            int track_phases = track_length - 1;
            int relative_poses_dimension = 7 * track_phases;

            Eigen::VectorXd relative_poses;
            if (feature_type == '1')
            {
                relative_poses = xk1k.tail(relative_poses_dimension);
            }
            else
            {
                relative_poses = xk1k.block(26, 0, relative_poses_dimension, 1);
            }

            Eigen::VectorXd relative_poses_to_first(relative_poses_dimension, 1);
            relative_poses_to_first.block(0, 0, 7, 1) << relative_poses.block(0, 0, 4, 1),
                -quatToRot(relative_poses.block(0, 0, 4, 1)) * relative_poses.block(4, 0, 3, 1);

            for (int i = 1; i < track_phases; ++i)
            {
                Eigen::Vector4d qI = quatMul(relative_poses.block(7 * i, 0, 4, 1),
                                             relative_poses_to_first.block(7 * (i - 1), 0, 4, 1));
                Eigen::Vector3d tI = quatToRot(relative_poses.block(7 * i, 0, 4, 1)) *
                                     (relative_poses_to_first.block(7 * (i - 1) + 4, 0, 3, 1) -
                                      relative_poses.block(7 * i + 4, 0, 3, 1));
                relative_poses_to_first.block(7 * i, 0, 7, 1) << qI, tI;
            }

            Eigen::VectorXd cam_relative_poses_to_first(relative_poses_dimension, 1);
            for (int i = 0; i < track_phases; ++i)
            {
                Eigen::Vector4d qC = rotToQuat(Rci_ * quatToRot(relative_poses_to_first.block(7 * i, 0, 4, 1)) * Ric_);
                Eigen::Vector3d tC = Rci_ * quatToRot(relative_poses_to_first.block(7 * i, 0, 4, 1)) * tic_ + Rci_ * relative_poses_to_first.block(7 * i + 4, 0, 3, 1) + tci_;
                cam_relative_poses_to_first.block(7 * i, 0, 7, 1) << qC, tC;
            }

            cv::Point2f pt_first = feature_measurements.front();
            feature_measurements.pop_front();

            double phi = std::atan2(pt_first.y, std::sqrt(std::pow(pt_first.x, 2) + 1));
            double psi = std::atan2(pt_first.x, 1);
            double rho = 0.0;

            if (std::fabs(phi) > 0.5 * 3.14 || std::fabs(psi) > 0.5 * 3.14)
            {
                ROS_DEBUG("Invalid inverse-depth feature estimate (0)!");
                continue;
            }

            Eigen::Vector3d epfinv;
            epfinv << std::cos(phi) * std::sin(psi), std::sin(phi), std::cos(phi) * std::cos(psi);

            Eigen::Matrix<double, 3, 2> Jang;
            Jang << -std::sin(phi) * std::sin(psi), std::cos(phi) * std::cos(psi),
                std::cos(phi), 0,
                -std::sin(phi) * std::cos(psi), -std::cos(phi) * std::sin(psi);

            Eigen::Matrix2d Rinv;
            Rinv << 1 / std::pow(sigma_image_noise_x_, 2), 0,
                0, 1 / std::pow(sigma_image_noise_y_, 2);

            int max_iter = 10;
            double lambda = 0.01;
            double last_cost = std::numeric_limits<double>::infinity();

            for (int iter = 0; iter < max_iter; ++iter)
            {
                Eigen::Matrix3d HTRinvH = Eigen::Matrix3d::Zero();
                Eigen::Vector3d HTRinve = Eigen::Vector3d::Zero();
                double cost = 0;

                Eigen::Vector3d h1 = epfinv;

                Eigen::Matrix<double, 2, 3> Hproj1;
                Hproj1 << 1 / h1(2), 0, -h1(0) / std::pow(h1(2), 2),
                    0, 1 / h1(2), -h1(1) / std::pow(h1(2), 2);

                Eigen::Matrix<double, 2, 3> H1;
                H1 << Hproj1 * Jang, Eigen::Vector2d::Zero();

                cv::Point2f pt1;
                pt1.x = h1(0) / h1(2);
                pt1.y = h1(1) / h1(2);

                Eigen::Vector2d e1;
                e1 << (pt_first - pt1).x, (pt_first - pt1).y;

                cost += e1.transpose() * Rinv * e1;

                HTRinvH.noalias() += H1.transpose() * Rinv * H1;
                HTRinve.noalias() += H1.transpose() * Rinv * e1;

                std::list<cv::Point2f>::const_iterator it = feature_measurements.begin();
                for (int i = 0; i < track_phases; ++i, ++it)
                {
                    Eigen::Matrix3d Rc = quatToRot(cam_relative_poses_to_first.block(7 * i, 0, 4, 1));
                    Eigen::Vector3d tc = cam_relative_poses_to_first.block(7 * i + 4, 0, 3, 1);
                    Eigen::Vector3d h = Rc * epfinv + rho * tc;

                    Eigen::Matrix<double, 2, 3> Hproj;
                    Hproj << 1 / h(2), 0, -h(0) / std::pow(h(2), 2),
                        0, 1 / h(2), -h(1) / std::pow(h(2), 2);

                    Eigen::Matrix<double, 2, 3> H;
                    H << Hproj * Rc * Jang, Hproj * tc;

                    cv::Point2f pt;
                    pt.x = h(0) / h(2);
                    pt.y = h(1) / h(2);

                    Eigen::Vector2d e;
                    e << ((*it) - pt).x, ((*it) - pt).y;

                    cost += e.transpose() * Rinv * e;

                    HTRinvH.noalias() += H.transpose() * Rinv * H;
                    HTRinve.noalias() += H.transpose() * Rinv * e;
                }

                if (cost <= last_cost)
                {
                    HTRinvH.diagonal() += lambda * HTRinvH.diagonal();
                    Eigen::Vector3d dpfinv = HTRinvH.colPivHouseholderQr().solve(HTRinve);

                    phi += dpfinv(0);
                    psi += dpfinv(1);
                    rho += dpfinv(2);

                    epfinv << std::cos(phi) * std::sin(psi), std::sin(phi), std::cos(phi) * std::cos(psi);

                    Jang << -std::sin(phi) * std::sin(psi), std::cos(phi) * std::cos(psi),
                        std::cos(phi), 0,
                        -std::sin(phi) * std::cos(psi), -std::cos(phi) * std::sin(psi);

                    if (std::fabs(last_cost - cost) < 1e-6 || dpfinv.norm() < 1e-6)
                        break;

                    lambda *= 0.1;
                    last_cost = cost;
                }
                else
                {
                    lambda *= 10;
                    last_cost = cost;
                }
            }

            if (std::fabs(phi) > 0.5 * 3.14 || std::fabs(psi) > 0.5 * 3.14 || std::isinf(rho) || rho < 0)
            {
                ROS_DEBUG("Invalid inverse-depth feature estimate (1)!");
                continue;
            }

            if (feature_type == '2')
            {
                track_length = std::ceil(0.5 * track_length);
                track_phases = track_length - 1;
            }

            Eigen::VectorXd tempr(2 * track_length, 1);
            Eigen::MatrixXd temp_Hx(2 * track_length, 6 * clone_states);
            Eigen::MatrixXd temp_Hf(2 * track_length, 3);
            tempr.setZero();
            temp_Hx.setZero();
            temp_Hf.setZero();

            int start_row = 0;
            int start_col;
            if (feature_type == '1')
            {
                start_col = 6 * (clone_states - track_phases);
            }
            else
            {
                start_col = 0;
            }

            Eigen::Vector3d h1 = epfinv;

            cv::Point2f pt1;
            pt1.x = h1(0) / h1(2);
            pt1.y = h1(1) / h1(2);

            Eigen::Matrix<double, 2, 3> Hproj1;
            Hproj1 << 1 / h1(2), 0, -h1(0) / std::pow(h1(2), 2),
                0, 1 / h1(2), -h1(1) / std::pow(h1(2), 2);

            cv::Point2f e1 = pt_first - pt1;
            tempr.block(0, 0, 2, 1) << e1.x, e1.y;

            Eigen::Matrix3d tempm = Eigen::Matrix3d::Zero();
            tempm.block<3, 2>(0, 0) = Jang;
            temp_Hf.block<2, 3>(0, 0) = Hproj1 * tempm;

            start_row += 2;

            std::list<cv::Point2f>::const_iterator it = feature_measurements.begin();
            for (int i = 1; i < track_length; ++i, ++it)
            {
                Eigen::Matrix3d R = quatToRot(relative_poses_to_first.block(7 * (i - 1), 0, 4, 1));

                Eigen::Matrix3d Rc = quatToRot(cam_relative_poses_to_first.block(7 * (i - 1), 0, 4, 1));
                Eigen::Vector3d tc = cam_relative_poses_to_first.block(7 * (i - 1) + 4, 0, 3, 1);
                Eigen::Vector3d h = Rc * epfinv + rho * tc;

                cv::Point2f pt;
                pt.x = h(0) / h(2);
                pt.y = h(1) / h(2);

                Eigen::Matrix<double, 2, 3> Hproj;
                Hproj << 1 / h(2), 0, -h(0) / std::pow(h(2), 2),
                    0, 1 / h(2), -h(1) / std::pow(h(2), 2);

                cv::Point2f e = (*it) - pt;
                tempr.block(2 * i, 0, 2, 1) << e.x, e.y;

                Eigen::Matrix3d R0T = quatToRot(relative_poses_to_first.block(0, 0, 4, 1)).transpose();
                Eigen::Vector3d t0 = relative_poses_to_first.block(4, 0, 3, 1);

                Eigen::Matrix3d dpx0 = skew(Ric_ * epfinv + rho * tic_ + rho * R0T * t0);
                Eigen::Matrix<double, 3, 6> subH;
                subH << dpx0 * R0T, -rho * Eigen::Matrix3d::Identity();

                temp_Hx.block(start_row, start_col, 2, 6) = Hproj * Rci_ * R * subH;

                for (int j = 1; j < i; ++j)
                {
                    Eigen::Matrix3d R1T = quatToRot(relative_poses_to_first.block(7 * j, 0, 4, 1)).transpose();
                    Eigen::Vector3d t1 = relative_poses_to_first.block(7 * j + 4, 0, 3, 1);
                    Eigen::Matrix3d R2T = quatToRot(relative_poses_to_first.block(7 * (j - 1), 0, 4, 1)).transpose();

                    Eigen::Matrix3d dpx = skew(Ric_ * epfinv + rho * tic_ + rho * R1T * t1);
                    subH << dpx * R1T, -rho * R2T;

                    temp_Hx.block(start_row, start_col + 6 * j, 2, 6) = Hproj * Rci_ * R * subH;
                }

                temp_Hf.block(start_row, 0, 2, 3) << Hproj * Rc * Jang, Hproj * tc;

                start_row += 2;
            }

            int M = start_row;
            int N = temp_Hf.cols();

            if (temp_Hf.col(N - 1).norm() < 1e-4)
            {
                ROS_DEBUG("Hf is rank deficient!");
                N--;
            }

            Eigen::JacobiRotation<double> temp_Hf_GR;

            for (int n = 0; n < N; ++n)
            {
                for (int m = M - 1; m > n; m--)
                {
                    temp_Hf_GR.makeGivens(temp_Hf(m - 1, n), temp_Hf(m, n));

                    (temp_Hf.block(m - 1, n, 2, N - n)).applyOnTheLeft(0, 1, temp_Hf_GR.adjoint());

                    (temp_Hx.block(m - 1, 0, 2, temp_Hx.cols())).applyOnTheLeft(0, 1, temp_Hf_GR.adjoint());

                    (tempr.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, temp_Hf_GR.adjoint());
                }
            }

            int DOF = M - N;
            Eigen::VectorXd _tempr = tempr.block(N, 0, DOF, 1);
            Eigen::MatrixXd _temp_Hx = temp_Hx.block(N, 0, DOF, temp_Hx.cols());

            Eigen::VectorXd temp_R;
            temp_R.setOnes(DOF, 1);
            temp_R *= std::pow(sigma_image_noise_x_ > sigma_image_noise_y_ ? sigma_image_noise_x_ : sigma_image_noise_y_, 2);

            Eigen::MatrixXd temp_S;
            temp_S = _temp_Hx * Pk1k.block(24, 24, 6 * clone_states, 6 * clone_states) * (_temp_Hx.transpose());
            temp_S.diagonal() += temp_R;
            temp_S = 0.5 * (temp_S + temp_S.transpose());

            double mahalanobis_dist = (_tempr.transpose() * (temp_S.colPivHouseholderQr().solve(_tempr))).norm();

            if (mahalanobis_dist < CHI_THRESHOLD[DOF - 1])
            {
                r.block(row_count, 0, DOF, 1) = _tempr;
                Hx.block(row_count, 24, DOF, 6 * clone_states) = _temp_Hx;

                row_count += DOF;
                good_feature_count++;

                if (rho > 0)
                {
                    Eigen::VectorXd posek = relative_poses_to_first.tail(7);
                    Eigen::MatrixXd Rk = quatToRot(posek.head(4));
                    Eigen::Vector3d tk = posek.tail(3);

                    Eigen::Vector3d pfc = 1 / rho * epfinv;
                    Eigen::Vector3d pf1 = Ric_ * pfc + tic_;
                    Eigen::Vector3d pfk = Rk * pf1 + tk;

                    geometry_msgs::Point feature;
                    feature.x = pfk(0);
                    feature.y = pfk(1);
                    feature.z = pfk(2);
                    cloud.points.push_back(feature);
                }
            }
            else
            {
                ROS_DEBUG("Failed in Mahalanobis distance test!");
                continue;
            }
        }

        feature_publish_.publish(cloud);

        if (good_feature_count > 2)
        {
            Eigen::VectorXd ro = r.block(0, 0, row_count, 1);
            Eigen::MatrixXd Ho = Hx.block(0, 0, row_count, Hx.cols());

            Eigen::VectorXd Ro;
            Ro.setOnes(row_count, 1);
            Ro *= std::pow(sigma_image_noise_x_ > sigma_image_noise_y_ ? sigma_image_noise_x_ : sigma_image_noise_y_, 2);

            Eigen::VectorXd rn;
            Eigen::MatrixXd Hn;
            Eigen::VectorXd Rn;

            if (Ho.rows() > Ho.cols() - 24)
            {
                int M = Ho.rows();
                int N = Ho.cols() - 24;

                Eigen::MatrixXd temp_Hw = Ho.block(0, 24, M, N);

                for (int i = N; i > 0; i--)
                {
                    if (temp_Hw.col(i - 1).norm() == 0)
                    {
                        ROS_DEBUG("Hw is rank deficient!");
                        N--;
                    }
                    else
                    {
                        break;
                    }
                }

                Eigen::JacobiRotation<double> temp_Hw_GR;

                for (int n = 0; n < N; ++n)
                {
                    for (int m = M - 1; m > n; m--)
                    {
                        temp_Hw_GR.makeGivens(temp_Hw(m - 1, n), temp_Hw(m, n));

                        (temp_Hw.block(m - 1, n, 2, N - n)).applyOnTheLeft(0, 1, temp_Hw_GR.adjoint());

                        (ro.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, temp_Hw_GR.adjoint());
                    }
                }

                Ho.block(0, 24, M, N) = temp_Hw.block(0, 0, M, N);

                int rank = 0;
                for (int i = 0; i < M; ++i)
                {
                    if (Ho.row(i).norm() < 1e-4)
                    {
                        break;
                    }
                    else
                    {
                        rank++;
                    }
                }

                rn = ro.block(0, 0, rank, 1);
                Hn = Ho.block(0, 0, rank, Ho.cols());
                Rn.setOnes(rank, 1);
                Rn *= std::pow(sigma_image_noise_x_ > sigma_image_noise_y_ ? sigma_image_noise_x_ : sigma_image_noise_y_, 2);
            }
            else
            {
                rn = ro;
                Hn = Ho;
                Rn = Ro;
            }

            Eigen::MatrixXd S = Hn * Pk1k * (Hn.transpose());
            S.diagonal() += Rn;
            S = 0.5 * (S + S.transpose());
            Eigen::MatrixXd K = Pk1k * (Hn.transpose()) * (S.inverse());
            Eigen::VectorXd dx = K * rn;

            xk1k1.resize(xk1k.rows(), 1);

            Eigen::Vector4d dqG;
            dqG(0) = 0.5 * dx(0);
            dqG(1) = 0.5 * dx(1);
            dqG(2) = 0.5 * dx(2);

            double dqGvn = (dqG.head(3)).norm();
            if (dqGvn < 1)
            {
                dqG(3) = std::sqrt(1 - std::pow(dqGvn, 2));
            }
            else
            {
                dqG.head(3) *= (1 / std::sqrt(1 + std::pow(dqGvn, 2)));
                dqG(3) = 1 / std::sqrt(1 + std::pow(dqGvn, 2));
            }

            xk1k1.block(0, 0, 4, 1) = quatMul(dqG, xk1k.block(0, 0, 4, 1));
            xk1k1.block(4, 0, 6, 1) = dx.block(3, 0, 6, 1) + xk1k.block(4, 0, 6, 1);

            Eigen::Vector3d g = xk1k1.block(7, 0, 3, 1);
            g.normalize();
            xk1k1.block(7, 0, 3, 1) = g;

            Eigen::Vector4d dqR;
            dqR(0) = 0.5 * dx(9);
            dqR(1) = 0.5 * dx(10);
            dqR(2) = 0.5 * dx(11);

            double dqRvn = (dqR.head(3)).norm();
            if (dqRvn < 1)
            {
                dqR(3) = std::sqrt(1 - std::pow(dqRvn, 2));
            }
            else
            {
                dqR.head(3) *= (1 / std::sqrt(1 + std::pow(dqRvn, 2)));
                dqR(3) = 1 / std::sqrt(1 + std::pow(dqRvn, 2));
            }

            xk1k1.block(10, 0, 4, 1) = quatMul(dqR, xk1k.block(10, 0, 4, 1));
            xk1k1.block(14, 0, 12, 1) = dx.block(12, 0, 12, 1) + xk1k.block(14, 0, 12, 1);

            for (int pose_idx = 0; pose_idx < clone_states; ++pose_idx)
            {
                Eigen::Vector4d dqc;
                dqc(0) = 0.5 * dx(24 + 6 * pose_idx);
                dqc(1) = 0.5 * dx(24 + 6 * pose_idx + 1);
                dqc(2) = 0.5 * dx(24 + 6 * pose_idx + 2);

                double dqcvn = (dqc.head(3)).norm();
                if (dqcvn < 1)
                {
                    dqc(3) = std::sqrt(1 - std::pow(dqcvn, 2));
                }
                else
                {
                    dqc.head(3) *= (1 / std::sqrt(1 + std::pow(dqcvn, 2)));
                    dqc(3) = 1 / std::sqrt(1 + std::pow(dqcvn, 2));
                }

                xk1k1.block(26 + 7 * pose_idx, 0, 4, 1) = quatMul(dqc, xk1k.block(26 + 7 * pose_idx, 0, 4, 1));
                xk1k1.block(26 + 7 * pose_idx + 4, 0, 3, 1) = dx.block(24 + 6 * pose_idx + 3, 0, 3, 1) + xk1k.block(26 + 7 * pose_idx + 4, 0, 3, 1);
            }

            Eigen::MatrixXd _I = Eigen::MatrixXd::Identity(Pk1k.rows(), Pk1k.cols());
            Eigen::MatrixXd I_KH = _I - K * Hn;

            Pk1k1 = I_KH * Pk1k * (I_KH.transpose());
            Pk1k1 += Rn(0) * K * (K.transpose());
            Pk1k1 = 0.5 * (Pk1k1 + Pk1k1.transpose());
        }
        else
        {
            ROS_DEBUG("Too few measurements for update!");

            xk1k1 = xk1k;
            Pk1k1 = Pk1k;
        }
    }
}