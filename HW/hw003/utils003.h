//
// Created by Volodymyr Avvakumov on 30.03.2026.
//

#ifndef CODE_UTILS003_H
#define CODE_UTILS003_H

#include <filesystem>
#include <vector>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/types.hpp>

static const std::string PATH_TO_DATA =
    (std::filesystem::path(__FILE__).parent_path() / "../../data/").lexically_normal().string()
    + "/";


std::vector<cv::Point2f> readPointsFromFile(const std::string& filename);

std::vector<cv::Point2d> points2f2points2d(const std::vector<cv::Point2f>& points);

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

std::tuple<size_t, size_t, size_t, size_t> get4RandomIndex(size_t n);

cv::Affine3d PnPbyIdx(
    const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_NED_X,
    const std::vector<size_t>& indices,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs
);

std::vector<size_t> getPnPInliersIdx(
    const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_X,
    double threshold_in_px,
    const cv::Matx33d& R,
    const cv::Vec3d& t,
    const cv::Matx33d& cameraMatrix
);

template<typename T>
std::vector<T> selectByIdx(std::vector<T> inp, std::vector<int> idxs);

std::vector<std::pair<double, double>> cvPoints2standardPoints(const std::vector<cv::Point2f>& points);cv::Affine3d RANSACforPnP(const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_X,
    double threshold_in_px,
    int max_iterations,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);



#endif //CODE_UTILS003_H