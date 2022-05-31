#include "rcvio/ransac.hpp"

#include <cmath>

#include <opencv2/core/eigen.hpp>

#include <rcvio/numerics.h>

namespace rcvio
{
    Ransac::Ransac(const cv::FileStorage &fs_settings)
    {
        const int use_sampson = fs_settings["Tracker.UseSampson"];
        use_sampson_ = use_sampson;

        inlier_threshold_ = fs_settings["Tracker.nInlierThreshold"];

        small_angle_ = fs_settings["IMU.nSmallAngle"];

        cv::Mat T(4, 4, CV_32F);
        fs_settings["Camera.T_BC0"] >> T;
        Eigen::Matrix4d Tic;
        cv::cv2eigen(T, Tic);
        Ric_ = Tic.block<3, 3>(0, 0);
        Rci_ = Ric_.transpose();
    }

    void Ransac::setPointPair(const int inlier_candidates, const int iterations)
    {
        std::vector<int> indices(inlier_candidates);

        for (int i = 0; i < inlier_candidates; ++i)
        {
            indices.at(i) = i;
        }

        int iter = 0;
        for (;;)
        {
            int index_A, index_B;
            do
            {
                index_A = std::rand() % inlier_candidates;
            } while (indices.at(index_A) == -1);

            do
            {
                index_B = std::rand() % inlier_candidates;
            } while (indices.at(index_B) == -1 || index_A == index_B);

            ransac_model_.two_points(iter, 0) = inlier_candidate_indices_.at(indices.at(index_A));
            ransac_model_.two_points(iter, 1) = inlier_candidate_indices_.at(indices.at(index_B));

            indices.at(index_A) = -1;
            indices.at(index_B) = -1;
            iter++;

            if (iter == iterations)
            {
                break;
            }
        }
    }

    void Ransac::setRansacModel(const Eigen::MatrixXd &points_1,
                                const Eigen::MatrixXd &points_2,
                                const Eigen::Matrix3d &R,
                                const int iter_num)
    {
        Eigen::Vector3d point_A1 = points_1.col(ransac_model_.two_points(iter_num, 0));
        Eigen::Vector3d point_A2 = points_2.col(ransac_model_.two_points(iter_num, 0));
        Eigen::Vector3d point_B1 = points_1.col(ransac_model_.two_points(iter_num, 1));
        Eigen::Vector3d point_B2 = points_2.col(ransac_model_.two_points(iter_num, 1));

        Eigen::Vector3d point_A0 = R * point_A1;
        Eigen::Vector3d point_B0 = R * point_B1;

        double c1 = point_A2(0) * point_A0(1) - point_A0(0) * point_A2(1);
        double c2 = point_A0(1) * point_A2(2) - point_A2(1) * point_A0(2);
        double c3 = point_A2(0) * point_A0(2) - point_A0(0) * point_A2(2);
        double c4 = point_B2(0) * point_B0(1) - point_B0(0) * point_B2(1);
        double c5 = point_B0(1) * point_B2(2) - point_B2(1) * point_B0(2);
        double c6 = point_B2(0) * point_B0(2) - point_B0(0) * point_B2(2);

        double alpha = std::atan2(c3 * c5 - c2 * c6, c1 * c6 - c3 * c4);
        double beta = std::atan2(-c3, c1 * std::sin(alpha) + c2 * std::cos(alpha));
        Eigen::Vector3d t = Eigen::Vector3d(std::sin(beta) * std::cos(alpha), std::cos(beta), -std::sin(beta) * std::sin(alpha));

        ransac_model_.hypotheses.block<3, 3>(3 * iter_num, 0) = skew(t) * R;
    }

    void Ransac::getRotation(std::list<ImuData *> &imu_data, Eigen::Matrix3d &R)
    {
        Eigen::Matrix3d temp_R;
        temp_R.setIdentity();

        Eigen::Matrix3d I;
        I.setIdentity();

        for (std::list<ImuData *>::const_iterator it = imu_data.begin(); it != imu_data.end(); ++it)
        {
            Eigen::Vector3d wm = (*it)->wm_;
            double dt = (*it)->dt_;

            bool is_small_angle = false;
            if (wm.norm() < small_angle_)
            {
                is_small_angle = true;
            }

            double w1 = wm.norm();
            double wdt = w1 * dt;
            Eigen::Matrix3d wx = skew(wm);
            Eigen::Matrix3d wx2 = wx * wx;

            Eigen::Matrix3d delta_R;
            if (is_small_angle)
            {
                delta_R = I - dt * wx + (0.5 * std::pow(dt, 2)) * wx2;
            }
            else
            {
                delta_R = I - (std::sin(wdt) / w1) * wx + ((1 - std::cos(wdt)) / std::pow(w1, 2)) * wx2;
            }
            assert(std::isnan(delta_R.norm()) != true);

            temp_R = delta_R * temp_R;
        }

        R = Rci_ * temp_R * Ric_;
    }

    void Ransac::countInliers(const Eigen::MatrixXd &points_1,
                              const Eigen::MatrixXd &points_2,
                              const int iter_num)

    {
        for (std::vector<int>::const_iterator it = inlier_candidate_indices_.begin(); it != inlier_candidate_indices_.end(); ++it)
        {
            int idx = *it;

            double distance;
            if (use_sampson_)
            {
                distance = sampsonError(points_1.col(idx), points_2.col(idx), ransac_model_.hypotheses.block<3, 3>(3 * iter_num, 0));
            }
            else
            {
                distance = algebraicError(points_1.col(idx), points_2.col(idx), ransac_model_.hypotheses.block<3, 3>(3 * iter_num, 0));
            }

            if (distance < inlier_threshold_)
            {
                ransac_model_.inliers(iter_num) += 1;
            }
        }
    }

    int Ransac::findInliers(const Eigen::MatrixXd &points_1,
                            const Eigen::MatrixXd &points_2,
                            std::list<ImuData *> &imu_data,
                            std::vector<unsigned char> &inlier_flag)
    {
        ransac_model_.hypotheses.setZero();
        ransac_model_.inliers.setZero();
        ransac_model_.two_points.setZero();

        inlier_candidate_indices_.clear();

        int inlier_candidates = 0;

        for (int i = 0; i < static_cast<int>(inlier_flag.size()); ++i)
        {
            if (inlier_flag.at(i))
            {
                inlier_candidate_indices_.push_back(i);
                inlier_candidates++;
            }
        }

        if (inlier_candidates > ransac_model_.iterations)
        {
            setPointPair(inlier_candidates, ransac_model_.iterations);
        }
        else
        {
            return 0;
        }

        Eigen::Matrix3d R;
        getRotation(imu_data, R);

        int winner_inliers_number = 0;
        int winner_hypothesis_idx = 0;
        for (int i = 0; i < ransac_model_.iterations; ++i)
        {
            setRansacModel(points_1, points_2, R, i);
            countInliers(points_1, points_2, i);

            if (ransac_model_.inliers(i) > winner_inliers_number)
            {
                winner_inliers_number = ransac_model_.inliers(i);
                winner_hypothesis_idx = i;
            }
        }

        Eigen::Matrix3d winner_E = ransac_model_.hypotheses.block<3, 3>(3 * winner_hypothesis_idx, 0);

        int new_outliers = 0;
        for (int i = 0; i < inlier_candidates; ++i)
        {
            int idx = inlier_candidate_indices_.at(i);

            double distance;
            if (use_sampson_)
            {
                distance = sampsonError(points_1.col(idx), points_2.col(idx), winner_E);
            }
            else
            {
                distance = algebraicError(points_1.col(idx), points_2.col(idx), winner_E);
            }

            if (distance > inlier_threshold_ || std::isnan(distance))
            {
                inlier_flag.at(idx) = 0;
                new_outliers++;
            }
        }

        return inlier_candidates - new_outliers;
    }

    double Ransac::sampsonError(const Eigen::Vector3d &pt_1,
                                const Eigen::Vector3d &pt_2,
                                const Eigen::Matrix3d &E) const
    {
        Eigen::Vector3d fx1 = E * pt_1;
        Eigen::Vector3d fx2 = E.transpose() * pt_2;

        return (std::pow(pt_2.transpose() * E * pt_1, 2)) /
               (std::pow(fx1(0), 2) + std::pow(fx1(1), 2) +
                std::pow(fx2(0), 2) + std::pow(fx2(1), 2));
    }

    double Ransac::algebraicError(const Eigen::Vector3d &pt_1,
                                  const Eigen::Vector3d &pt_2,
                                  const Eigen::Matrix3d &E) const
    {
        return std::fabs(pt_2.transpose() * E * pt_1);
    }
}