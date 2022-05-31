#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>

#include <opencv2/core/core.hpp>

#include "rcvio/system.hpp"

class ImageGrabber
{
public:
    ImageGrabber(rcvio::System *sys) : sys_(sys) {}

    void grabImage(const sensor_msgs::ImageConstPtr &msg);

public:
    rcvio::System *sys_;
};

class ImuGrabber
{
public:
    ImuGrabber(rcvio::System *sys) : sys_(sys) {}

    void grabImu(const sensor_msgs::ImuConstPtr &msg);

public:
    rcvio::System *sys_;
};

void ImageGrabber::grabImage(const sensor_msgs::ImageConstPtr &msg)
{
    static int last_seq = -1;
    if ((int)msg->header.seq != last_seq + 1 && last_seq != -1)
    {
        ROS_DEBUG("Image message drop! curr seq: %d expected seq: %d.", msg->header.seq, last_seq + 1);
    }
    last_seq = msg->header.seq;

    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    rcvio::ImageData *data = new rcvio::ImageData();
    data->image_ = cv_ptr->image.clone();
    data->ts_ = cv_ptr->header.stamp.toSec();

    sys_->pushImageData(data);

    sys_->monoVIO();
}

void ImuGrabber::grabImu(const sensor_msgs::ImuConstPtr &msg)
{
    static int last_seq = -1;

    if ((int)msg->header.seq != last_seq + 1 && last_seq != -1)
    {
        ROS_DEBUG("IMU message drop! curr seq: %d expected seq: %d.", msg->header.seq, last_seq + 1);
    }
    last_seq = msg->header.seq;

    Eigen::Vector3d wm;
    tf::vectorMsgToEigen(msg->angular_velocity, wm);

    Eigen::Vector3d am;
    tf::vectorMsgToEigen(msg->linear_acceleration, am);

    double curr_time = msg->header.stamp.toSec();

    rcvio::ImuData *data = new rcvio::ImuData();
    data->wm_ = wm;
    data->am_ = am;
    data->ts_ = curr_time;

    static double last_time = -1;
    if (last_time != -1)
    {
        data->dt_ = curr_time - last_time;
    }
    else
    {
        data->dt_ = 0;
    }
    last_time = curr_time;

    sys_->pushImuData(data);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rcvio");
    ros::start();

    rcvio::System sys(argv[1]);

    ImageGrabber igb1(&sys);
    ImuGrabber igb2(&sys);

    ros::NodeHandle node_handler;
    ros::Subscriber image_sub = node_handler.subscribe("/camera/image_raw", 10, &ImageGrabber::grabImage, &igb1);
    ros::Subscriber imu_sub = node_handler.subscribe("/imu", 100, &ImuGrabber::grabImu, &igb2);

    ros::spin();

    ros::shutdown();

    return 0;
}