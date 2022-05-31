#pragma once

#include <list>
#include <mutex>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "rcvio/macros.h"

namespace rcvio
{
    struct ImuData
    {
        Eigen::Vector3d wm_;
        Eigen::Vector3d am_;
        double ts_;
        double dt_;

        ImuData()
        {
            wm_.setZero();
            am_.setZero();
            ts_ = 0;
            dt_ = 0;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct ImageData
    {
        cv::Mat image_;
        double ts_;

        ImageData()
        {
            image_ = cv::Mat();
            ts_ = 0;
        }
    };

    class InputBuffer
    {
    public:
        POINTER_TYPEDEFS(InputBuffer);

        InputBuffer() = default;

        void pushImuData(ImuData *data);
        void pushImageData(ImageData *data);

        bool getMeasurements(double time_offset, std::pair<ImageData *, std::list<ImuData *>> &measurements);

    protected:
        std::list<ImuData *> imu_fifo_;
        std::list<ImageData *> image_fifo_;

        std::mutex mutex_;
    };
}