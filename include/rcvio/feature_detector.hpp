#pragma once

#include <deque>
#include <vector>

#include <opencv2/core/core.hpp>

#include "rcvio/macros.h"

namespace rcvio
{

    class FeatureDetector
    {
    public:
        POINTER_TYPEDEFS(FeatureDetector);

        FeatureDetector(const cv::FileStorage &fs_settings);

        int detect(const cv::Mat &image,
                   const int max_corners,
                   const int scale,
                   std::vector<cv::Point2f> &corners);

        int findNewer(const std::vector<cv::Point2f> &corners,
                      const std::vector<cv::Point2f> &ref_corners,
                      std::deque<cv::Point2f> &new_corners);

    private:
        void chessGrid(const std::vector<cv::Point2f> &corners);

    private:
        int image_cols_;
        int image_rows_;

        float min_distance_;
        float quality_level_;

        int grid_cols_;
        int grid_rows_;

        int offset_x_;
        int offset_y_;

        int blocks_;
        float block_size_x_;
        float block_size_y_;
        int max_features_per_block_;

        std::vector<std::vector<cv::Point2f>> grid_;
    };
}