#pragma once

#include <list>

#include <Eigen/Core>

#include "rcvio/macros.h"
#include "rcvio/input_buffer.hpp"

namespace rcvio
{
    class PreIntegrator
    {
    public:
        POINTER_TYPEDEFS(PreIntegrator);
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PreIntegrator(const cv::FileStorage &fs_settings);

        void propagate(Eigen::VectorXd &xkk,
                       Eigen::MatrixXd &Pkk,
                       std::list<ImuData *> &imu_data);

    public:
        Eigen::VectorXd xk1k;
        Eigen::MatrixXd Pk1k;

    private:
        double gravity_;
        double small_angle_;

        double sigma_gyro_noise_;
        double sigma_accel_noise_;
        double sigma_gyro_bias_;
        double sigma_accel_bias_;

        Eigen::Matrix<double, 12, 12> sigma_;
    };
}