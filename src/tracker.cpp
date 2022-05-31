#include "rcvio/tracker.hpp"

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sensor_msgs/Image.h>

namespace rcvio
{
    cv::Scalar red = CV_RGB(255, 64, 64);
    cv::Scalar green = CV_RGB(64, 255, 64);
    cv::Scalar blue = CV_RGB(64, 64, 255);

    Tracker::Tracker(const cv::FileStorage &fs_settings)
    {
        const float fx = fs_settings["Camera.fx"];
        const float fy = fs_settings["Camera.fy"];
        const float cx = fs_settings["Camera.cx"];
        const float cy = fs_settings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(K_);

        cv::Mat D(4, 1, CV_32F);
        D.at<float>(0) = fs_settings["Camera.k1"];
        D.at<float>(1) = fs_settings["Camera.k2"];
        D.at<float>(2) = fs_settings["Camera.p1"];
        D.at<float>(3) = fs_settings["Camera.p2"];
        const float k3 = fs_settings["Camera.k3"];
        if (k3 != 0)
        {
            D.resize(5);
            D.at<float>(4) = k3;
        }
        D.copyTo(D_);

        const int is_rgb = fs_settings["Camera.RGB"];
        is_rgb_ = is_rgb;

        const int is_fisheye = fs_settings["Camera.Fisheye"];
        is_fisheye_ = is_fisheye;

        const int enable_equalizer = fs_settings["Tracker.EnableEqualizer"];
        enable_equalizer_ = enable_equalizer;

        max_features_per_image_ = fs_settings["Tracker.nFeatures"];
        max_features_for_update_ = std::ceil(0.5 * max_features_per_image_);

        tracking_history_.resize(max_features_per_image_);

        max_tracking_length_ = fs_settings["Tracker.nMaxTrackingLength"];
        min_tracking_length_ = fs_settings["Tracker.nMinTrackingLength"];

        is_the_first_image_ = true;

        last_image_ = cv::Mat();

        feature_detector_.reset(new FeatureDetector(fs_settings));
        ransac_.reset(new Ransac(fs_settings));

        track_publish_ = tracker_node_.advertise<sensor_msgs::Image>("/rcvio/track", 1);
        newer_publish_ = tracker_node_.advertise<sensor_msgs::Image>("/rcvio/newer", 1);
    }

    void Tracker::displayTrack(const cv::Mat &image_in,
                               std::vector<cv::Point2f> &points_1,
                               std::vector<cv::Point2f> &points_2,
                               std::vector<unsigned char> &inlier_flag,
                               cv_bridge::CvImage &image_out)
    {
        image_out.header = std_msgs::Header();
        image_out.encoding = "bgr8";

        cv::cvtColor(image_in, image_out.image, CV_GRAY2BGR);

        for (int i = 0; i < static_cast<int>(points_1.size()); ++i)
        {
            if (inlier_flag.at(i) != 0)
            {
                cv::circle(image_out.image, points_1.at(i), 3, blue, -1);
                cv::line(image_out.image, points_1.at(i), points_2.at(i), blue);
            }
            else
            {
                cv::circle(image_out.image, points_1.at(i), 3, red, 0);
            }
        }
    }

    void Tracker::displayNewer(const cv::Mat &image_in,
                               std::vector<cv::Point2f> &features,
                               std::deque<cv::Point2f> &new_features,
                               cv_bridge::CvImage &image_out)
    {
        image_out.header = std_msgs::Header();
        image_out.encoding = "bgr8";

        cv::cvtColor(image_in, image_out.image, CV_GRAY2BGR);

        for (int i = 0; i < static_cast<int>(features.size()); ++i)
        {
            cv::circle(image_out.image, features.at(i), 3, blue, 0);
        }
        for (int i = 0; i < static_cast<int>(new_features.size()); ++i)
        {
            cv::circle(image_out.image, new_features.at(i), 3, green, -1);
        }
    }

    void Tracker::track(const cv::Mat &image, std::list<ImuData *> &imu_data)
    {
        if (image.channels() == 3)
        {
            if (is_rgb_)
                cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
            else
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }
        else if (image.channels() == 4)
        {
            if (is_rgb_)
                cv::cvtColor(image, image, cv::COLOR_RGBA2GRAY);
            else
                cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
        }

        if (enable_equalizer_)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(5, 5));
            clahe->apply(image, image);
        }

        if (is_the_first_image_)
        {
            num_features_to_track_ = feature_detector_->detect(image, max_features_per_image_, 1, features_to_track_);

            if (num_features_to_track_ == 0)
            {
                ROS_DEBUG("No features available to track.");
                return;
            }

            std::vector<cv::Point2f> features_undistort_norm;
            undistort(num_features_to_track_, features_to_track_, features_undistort_norm);

            points_1_for_ransac_.setZero(3, num_features_to_track_);
            for (int i = 0; i < num_features_to_track_; ++i)
            {
                cv::Point2f pt_un = features_undistort_norm.at(i);
                tracking_history_.at(i).push_back(pt_un);

                Eigen::Vector3d pt_un_e = Eigen::Vector3d(pt_un.x, pt_un.y, 1);
                points_1_for_ransac_.block(0, i, 3, 1) = pt_un_e;

                inlier_indices_.push_back(i);
            }

            for (int i = num_features_to_track_; i < max_features_per_image_; ++i)
            {
                free_indices_.push_back(i);
            }

            is_the_first_image_ = false;
        }
        else
        {
            cv::Size win_size(15, 15);
            cv::TermCriteria term_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-2);
            std::vector<cv::Point2f> features_tracked;
            std::vector<unsigned char> inlier_flag;
            std::vector<float> tracking_error;

            cv::calcOpticalFlowPyrLK(last_image_,
                                     image,
                                     features_to_track_,
                                     features_tracked,
                                     inlier_flag,
                                     tracking_error,
                                     win_size,
                                     3,
                                     term_criteria,
                                     0,
                                     1e-3);

            if (features_tracked.empty())
            {
                ROS_DEBUG("No features tracked in current image.");
                return;
            }

            std::vector<cv::Point2f> features_undistort_norm;
            undistort(num_features_to_track_, features_tracked, features_undistort_norm);

            points_2_for_ransac_.setZero(3, num_features_to_track_);
            for (int i = 0; i < num_features_to_track_; ++i)
            {
                cv::Point2f pt_un = features_undistort_norm.at(i);
                Eigen::Vector3d pt_un_e = Eigen::Vector3d(pt_un.x, pt_un.y, 1);
                points_2_for_ransac_.block(0, i, 3, 1) = pt_un_e;
            }

            ransac_->findInliers(points_1_for_ransac_, points_2_for_ransac_, imu_data, inlier_flag);

            cv_bridge::CvImage image_track;
            displayTrack(image, features_to_track_, features_tracked, inlier_flag, image_track);
            track_publish_.publish(image_track.toImageMsg());

            feature_types_for_update_.clear();
            feature_measurements_for_update_.clear();
            feature_measurements_for_update_.resize(max_features_for_update_);

            features_to_track_.clear();
            std::vector<int> inlier_indices_to_track;
            Eigen::MatrixXd temp_points_for_ransac(3, max_features_per_image_);

            int measurements_count = 0;
            int inlier_count = 0;

            for (int i = 0; i < num_features_to_track_; ++i)
            {
                if (!inlier_flag.at(i))
                {
                    int idx = inlier_indices_.at(i);
                    free_indices_.push_back(idx);

                    if (static_cast<int>(tracking_history_.at(idx).size() >= min_tracking_length_))
                    {
                        if (measurements_count < max_features_for_update_)
                        {
                            feature_types_for_update_.push_back('1');
                            feature_measurements_for_update_.at(measurements_count) = tracking_history_.at(idx);
                            measurements_count++;
                        }
                    }

                    tracking_history_.at(idx).clear();
                }
            }

            for (int i = 0; i < num_features_to_track_; ++i)
            {
                if (inlier_flag.at(i))
                {
                    int idx = inlier_indices_.at(i);
                    inlier_indices_to_track.push_back(idx);

                    cv::Point2f pt = features_tracked.at(i);
                    features_to_track_.push_back(pt);

                    cv::Point2f pt_un = features_undistort_norm.at(i);
                    if (static_cast<int>(tracking_history_.at(idx).size()) == max_tracking_length_)
                    {
                        if (measurements_count < max_features_for_update_)
                        {
                            feature_types_for_update_.push_back('2');
                            feature_measurements_for_update_.at(measurements_count) = tracking_history_.at(idx);

                            while (tracking_history_.at(idx).size() > max_tracking_length_ - (std::ceil(0.5 * max_tracking_length_) - 1))
                                tracking_history_.at(idx).pop_front();

                            measurements_count++;
                        }
                        else
                        {
                            tracking_history_.at(idx).pop_front();
                        }
                    }
                    tracking_history_.at(idx).push_back(pt_un);

                    Eigen::Vector3d pt_un_e = Eigen::Vector3d(pt_un.x, pt_un.y, 1);
                    temp_points_for_ransac.block(0, inlier_count, 3, 1) = pt_un_e;

                    inlier_count++;
                }
            }

            if (!free_indices_.empty())
            {
                std::vector<cv::Point2f> temp_features;
                std::deque<cv::Point2f> new_features;

                feature_detector_->detect(image, max_features_per_image_, 2, temp_features);
                int num_new_features = feature_detector_->findNewer(temp_features, features_to_track_, new_features);

                cv_bridge::CvImage image_newer;
                displayNewer(image, temp_features, new_features, image_newer);
                newer_publish_.publish(image_newer.toImageMsg());

                if (num_new_features != 0)
                {
                    std::deque<cv::Point2f> new_features_undistort_norm;
                    undistort(num_new_features, new_features, new_features_undistort_norm);

                    for (;;)
                    {
                        int idx = free_indices_.front();
                        inlier_indices_to_track.push_back(idx);

                        cv::Point2f pt = new_features.front();
                        features_to_track_.push_back(pt);

                        cv::Point2f pt_un = new_features_undistort_norm.front();
                        tracking_history_.at(idx).push_back(pt_un);

                        Eigen::Vector3d pt_un_e = Eigen::Vector3d(pt_un.x, pt_un.y, 1);
                        temp_points_for_ransac.block(0, inlier_count, 3, 1) = pt_un_e;

                        inlier_count++;

                        free_indices_.pop_front();
                        new_features.pop_front();
                        new_features_undistort_norm.pop_front();

                        if (free_indices_.empty() || new_features.empty() || new_features_undistort_norm.empty() || inlier_count == max_features_per_image_)
                            break;
                    }
                }
            }

            num_features_to_track_ = inlier_count;
            inlier_indices_ = inlier_indices_to_track;
            points_1_for_ransac_ = temp_points_for_ransac.block(0, 0, 3, inlier_count);
        }

        image.copyTo(last_image_);
    }
}