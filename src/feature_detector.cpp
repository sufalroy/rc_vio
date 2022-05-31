#include "rcvio/feature_detector.hpp"

#include <opencv2/opencv.hpp>

namespace rcvio
{
    FeatureDetector::FeatureDetector(const cv::FileStorage &fs_settings)
    {
        min_distance_ = fs_settings["Tracker.nMinDist"];
        quality_level_ = fs_settings["Tracker.nQualLvl"];

        block_size_x_ = fs_settings["Tracker.nBlockSizeX"];
        block_size_y_ = fs_settings["Tracker.nBlockSizeY"];

        image_cols_ = fs_settings["Camera.width"];
        image_rows_ = fs_settings["Camera.height"];

        grid_cols_ = std::floor(image_cols_ / block_size_x_);
        grid_rows_ = std::floor(image_rows_ / block_size_y_);

        blocks_ = grid_cols_ * grid_rows_;

        offset_x_ = 0.5 * (image_cols_ - grid_cols_ * block_size_x_);
        offset_y_ = 0.5 * (image_rows_ - grid_rows_ * block_size_y_);

        int max_features_per_image = fs_settings["Tracker.nFeatures"];
        max_features_per_block_ = static_cast<float>(max_features_per_image) / blocks_;

        grid_.resize(blocks_);
    }

    int FeatureDetector::detect(const cv::Mat &image,
                                const int max_corners,
                                const int scale,
                                std::vector<cv::Point2f> &corners)
    {
        corners.clear();
        corners.reserve(max_corners);

        cv::goodFeaturesToTrack(image, corners, max_corners, quality_level_, scale * min_distance_);

        if (!corners.empty())
        {
            cv::Size sub_pix_win_size(std::floor(0.5 * min_distance_), std::floor(0.5 * min_distance_));
            cv::Size sub_pix_zero_zone(-1, -1);
            cv::TermCriteria sub_pix_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-2);
            cv::cornerSubPix(image, corners, sub_pix_win_size, sub_pix_zero_zone, sub_pix_criteria);
        }

        return static_cast<int>(corners.size());
    }

    void FeatureDetector::chessGrid(const std::vector<cv::Point2f> &corners)
    {
        grid_.clear();
        grid_.resize(blocks_);

        for (const cv::Point2f &pt : corners)
        {
            if (pt.x <= offset_x_ || pt.y <= offset_y_ ||
                pt.x >= (image_cols_ - offset_x_) || pt.y >= (image_rows_ - offset_y_))
            {
                continue;
            }

            int col = std::floor((pt.x - offset_x_) / block_size_x_);
            int row = std::floor((pt.y - offset_y_) / block_size_y_);

            int block_idx = row * grid_cols_ + col;
            grid_.at(block_idx).emplace_back(pt);
        }
    }

    int FeatureDetector::findNewer(const std::vector<cv::Point2f> &corners,
                                   const std::vector<cv::Point2f> &ref_corners,
                                   std::deque<cv::Point2f> &new_corners)
    {
        chessGrid(ref_corners);

        for (const cv::Point2f &pt : corners)
        {
            if (pt.x <= offset_x_ || pt.y <= offset_y_ ||
                pt.x >= (image_cols_ - offset_x_) || pt.y >= (image_rows_ - offset_y_))
            {
                continue;
            }

            int col = std::floor((pt.x - offset_x_) / block_size_x_);
            int row = std::floor((pt.y - offset_y_) / block_size_y_);

            float xl = col * block_size_x_ + offset_x_;
            float xr = xl + block_size_x_;
            float yt = row * block_size_y_ + offset_y_;
            float yb = yt + block_size_y_;

            if (std::fabs(pt.x - xl) < min_distance_ || std::fabs(pt.x - xr) < min_distance_ || std::fabs(pt.y - yt) < min_distance_ || std::fabs(pt.y - yb) < min_distance_)
            {
                continue;
            }

            int block_idx = row * grid_cols_ + col;

            if (static_cast<float>(grid_.at(block_idx).size()) < 0.75 * max_features_per_block_)
            {
                if (!grid_.at(block_idx).empty())
                {
                    int count = 0;

                    for (const cv::Point2f &bpt : grid_.at(block_idx))
                    {
                        if (cv::norm(pt - bpt) > 1 * min_distance_)
                        {
                            count++;
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (count == static_cast<int>(grid_.at(block_idx).size()))
                    {
                        new_corners.push_back(pt);
                        grid_.at(block_idx).push_back(pt);
                    }
                }
                else
                {
                    new_corners.push_back(pt);
                    grid_.at(block_idx).push_back(pt);
                }
            }
        }

        return static_cast<int>(new_corners.size());
    }
}