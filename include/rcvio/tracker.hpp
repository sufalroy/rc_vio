#pragma once

#include <list>
#include <deque>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "rcvio/ransac.hpp"
#include "rcvio/feature_detector.hpp"

namespace rcvio
{
    class Tracker
    {
    public:
        POINTER_TYPEDEFS(Tracker);
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Tracker(const cv::FileStorage &fs_settings);

        void track(const cv::Mat &image, std::list<ImuData *> &imu_data);

    private:
        template <typename T1, typename T2>
        void undistort(const int N, T1 &src, T2 &dst);

        void displayTrack(const cv::Mat &image_in,
                          std::vector<cv::Point2f> &points_1,
                          std::vector<cv::Point2f> &points_2,
                          std::vector<unsigned char> &inlier_flag,
                          cv_bridge::CvImage &image_out);

        void displayNewer(const cv::Mat &image_in,
                          std::vector<cv::Point2f> &features,
                          std::deque<cv::Point2f> &new_features,
                          cv_bridge::CvImage &image_out);

    public:
        std::vector<unsigned char> feature_types_for_update_;
        std::vector<std::list<cv::Point2f>> feature_measurements_for_update_;

    private:
        int max_features_per_image_;
        int max_features_for_update_;

        int max_tracking_length_;
        int min_tracking_length_;

        bool is_rgb_;
        bool is_fisheye_;
        bool is_the_first_image_;

        bool enable_equalizer_;

        cv::Mat K_;
        cv::Mat D_;

        cv::Mat last_image_;

        std::vector<std::list<cv::Point2f>> tracking_history_;

        std::vector<int> inlier_indices_;

        std::list<int> free_indices_;

        std::vector<cv::Point2f> features_to_track_;
        int num_features_to_track_;

        Eigen::MatrixXd points_1_for_ransac_;
        Eigen::MatrixXd points_2_for_ransac_;

        FeatureDetector::Ptr feature_detector_;
        Ransac::Ptr ransac_;

        ros::NodeHandle tracker_node_;
        ros::Publisher track_publish_;
        ros::Publisher newer_publish_;
    };

    template <typename T1, typename T2>
    void Tracker::undistort(const int N, T1 &src, T2 &dst)
    {
        cv::Mat mat(N, 2, CV_32F);

        for (int i = 0; i < N; i++)
        {
            mat.at<float>(i, 0) = src.at(i).x;
            mat.at<float>(i, 1) = src.at(i).y;
        }

        mat = mat.reshape(2);
        if (!is_fisheye_)
        {
            cv::undistortPoints(mat, mat, K_, D_);
        }
        else
        {
            cv::fisheye::undistortPoints(mat, mat, K_, D_);
        }
        mat = mat.reshape(1);

        dst.clear();
        for (int i = 0; i < N; ++i)
        {
            cv::Point2f pt_un;
            pt_un.x = mat.at<float>(i, 0);
            pt_un.y = mat.at<float>(i, 1);

            dst.push_back(pt_un);
        }
    }
}