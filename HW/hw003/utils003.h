//
// Created by Volodymyr Avvakumov on 30.03.2026.
//

#ifndef CODE_UTILS003_H
#define CODE_UTILS003_H

#include <filesystem>
#include <vector>
#include <opencv2/core/types.hpp>

static const std::string PATH_TO_DATA =
    (std::filesystem::path(__FILE__).parent_path() / "../../data/").lexically_normal().string()
    + "/";


std::vector<cv::Point2f> readPointsFromFile(const std::string& filename);

template<typename T>
std::pair<T, T> pick2(const std::vector<T>& v, size_t n);

std::pair<size_t,size_t> get2RandomIndex(size_t n);

double distance(const cv::Point2f& point1_line,
                const cv::Point2f& point2_line,
                const cv::Point2f& distant_point);

std::vector<cv::Point2f> getInliers(cv::Point2f line_point_1, cv::Point2f line_point_2, double inlierThreshold, const std::vector<cv::Point2f>& points);

cv::Point2d linearRegression(const std::vector<cv::Point2f>& points);

cv::Point2d RANSACforLinearRegression(const std::vector<cv::Point2f>& points, double inlierThreshold, size_t maxIterations);

std::vector<cv::Point2f> standardPoints2cvPoints(const std::vector<std::pair<double, double>>& points);

std::vector<std::pair<double, double>> cvPoints2standardPoints(const std::vector<cv::Point2f>& points);
#endif //CODE_UTILS003_H