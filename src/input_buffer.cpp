#include "rcvio/input_buffer.hpp"

namespace rcvio
{
    void InputBuffer::pushImuData(ImuData *data)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        imu_fifo_.push_back(data);
        if (!imu_fifo_.empty())
        {
            imu_fifo_.sort([](const ImuData *a, const ImuData *b)
                           { return a->ts_ < b->ts_; });
        }
    }

    void InputBuffer::pushImageData(ImageData *data)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        image_fifo_.push_back(data);
        if (!image_fifo_.empty())
        {
            image_fifo_.sort([](const ImageData *a, const ImageData *b)
                             { return a->ts_ < b->ts_; });
        }
    }

    bool InputBuffer::getMeasurements(double time_offset, std::pair<ImageData *, std::list<ImuData *>> &measurements)
    {
        if (imu_fifo_.empty() || image_fifo_.empty())
        {
            return false;
        }

        if (imu_fifo_.back()->ts_ < image_fifo_.front()->ts_ + time_offset)
        {
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);

        ImageData *image = image_fifo_.front();
        image_fifo_.pop_front();

        std::list<ImuData *> imus;

        while (!imu_fifo_.empty() && imu_fifo_.front()->ts_ <= image->ts_ + time_offset)
        {
            imus.push_back(imu_fifo_.front());
            imu_fifo_.pop_front();
        }

        if (imus.size() < 2)
        {
            return false;
        }

        measurements = {image, imus};

        return true;
    }
}