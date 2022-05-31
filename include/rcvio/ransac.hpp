#pragma once

#include <list>
#include <vector>

#include <Eigen/Core>

#include "rcvio/input_buffer.hpp"

namespace rcvio
{
    struct RansacModel
    {
        Eigen::MatrixXd hypotheses;
        Eigen::MatrixXi inliers;
        Eigen::MatrixXi two_points;
        int iterations;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RansacModel()
        {
            iterations = 16;
            hypotheses.resize(iterations * 3, 3);
            inliers.resize(iterations, 1);
            two_points.resize(iterations, 2);
        }
    };

    class Ransac
    {
    public:
        POINTER_TYPEDEFS(Ransac);
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Ransac(const cv::FileStorage &fs_settings);

        void setPointPair(const int inlier_candidates, const int iterations);

        void setRansacModel(const Eigen::MatrixXd &points_1,
                            const Eigen::MatrixXd &points_2,
                            const Eigen::Matrix3d &R,
                            const int iter_num);

        void getRotation(std::list<ImuData *> &imu_data, Eigen::Matrix3d &R);

        void countInliers(const Eigen::MatrixXd &points_1,
                          const Eigen::MatrixXd &points_2,
                          const int iter_num);

        int findInliers(const Eigen::MatrixXd &points_1,
                        const Eigen::MatrixXd &points_2,
                        std::list<ImuData *> &imu_data,
                        std::vector<unsigned char> &inlier_flag);

        double sampsonError(const Eigen::Vector3d &pt_1,
                            const Eigen::Vector3d &pt_2,
                            const Eigen::Matrix3d &E) const;

        double algebraicError(const Eigen::Vector3d &pt_1,
                              const Eigen::Vector3d &pt_2,
                              const Eigen::Matrix3d &E) const;

    private:
        bool use_sampson_;
        double inlier_threshold_;

        double small_angle_;

        Eigen::Matrix3d Rci_;
        Eigen::Matrix3d Ric_;

        RansacModel ransac_model_;

        std::vector<int> inlier_candidate_indices_;
    };
}