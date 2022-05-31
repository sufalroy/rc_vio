#pragma once

#include <string>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <tf/transform_broadcaster.h>

#include "rcvio/tracker.hpp"
#include "rcvio/updater.hpp"
#include "rcvio/input_buffer.hpp"
#include "rcvio/pre_integrator.hpp"

namespace rcvio
{
    class System
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        System(const std::string &settings_file);

        inline void pushImuData(ImuData *data) { input_buffer_->pushImuData(data); }
        inline void pushImageData(ImageData *data) { input_buffer_->pushImageData(data); }

        void initialize(const Eigen::Vector3d &w, const Eigen::Vector3d &a,
                        const int imu_data, const bool enable_alignment);

        void monoVIO();

    private:
        double imu_rate_;
        double cam_time_offset_;

        int sliding_window_size_;
        int min_clone_states_;

        bool enable_alignment_;
        bool record_outputs_;

        bool is_moving_;
        bool is_ready_;

        double threshold_angle_;
        double threshold_displacement_;

        double gravity_;

        double sigma_gyro_noise_;
        double sigma_gyro_bias_;
        double sigma_accel_noise_;
        double sigma_accel_bias_;

        Eigen::VectorXd xkk;
        Eigen::MatrixXd Pkk;

        Tracker::Ptr tracker_;
        Updater::Ptr updater_;
        InputBuffer::Ptr input_buffer_;
        PreIntegrator::Ptr pre_integrator_;

        ros::NodeHandle system_node_;
        ros::Publisher path_publish_;
        ros::Publisher odometry_publish_;
        tf::TransformBroadcaster tf_publish_;
    };
}