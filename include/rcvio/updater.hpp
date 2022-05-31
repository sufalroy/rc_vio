#pragma once

#include <list>
#include <vector>

#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>

#include <ros/ros.h>

#include "rcvio/macros.h"

namespace rcvio
{
    class Updater
    {
    public:
        POINTER_TYPEDEFS(Updater);
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Updater(const cv::FileStorage &fs_settings);

        void update(Eigen::VectorXd &xk1k,
                    Eigen::MatrixXd &Pk1k,
                    std::vector<unsigned char> &feature_types_for_update,
                    std::vector<std::list<cv::Point2f>> &feature_measurements_for_update);

    public:
        Eigen::VectorXd xk1k1;
        Eigen::MatrixXd Pk1k1;

    private:
        double cam_rate_;

        double sigma_image_noise_x_;
        double sigma_image_noise_y_;

        Eigen::Matrix3d Ric_;
        Eigen::Vector3d tic_;
        Eigen::Matrix3d Rci_;
        Eigen::Vector3d tci_;

        ros::NodeHandle updater_node_;
        ros::Publisher feature_publish_;
        double publish_rate_;
    };
}