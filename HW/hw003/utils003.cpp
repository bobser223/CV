//
// Created by Volodymyr Avvakumov on 30.03.2026.
//

#include "utils003.h"


#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>

std::vector<cv::Point2f> readPointsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<cv::Point2f> points;
    float x, y;
    char comma;

    while (file >> x >> comma >> y) {
        if (comma != ',') {
            throw std::runtime_error("Invalid format in file: expected comma");
        }
        points.emplace_back(x, y);
    }

    return points;
}

std::vector<cv::Point2d> points2f2points2d(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2d> points2d;
    for (const auto& point : points) {
        points2d.emplace_back(point.x, point.y);
    }
    return points2d;
}

template<typename T>
std::pair<T, T> pick2(const std::vector<T>& v, size_t n) {

    static std::mt19937 random(123456);

    int i = random() % n;
    int j = random() % (n - 1);

    if (j >= i) j++;  //j != i

    return {v[i], v[j]};
}

std::pair<size_t,size_t> get2RandomIndex(size_t n) {
    static std::mt19937 random(123456);
    int i = random() % n;
    int j = random() % (n - 1);

    if (j >= i) j++;  //j != i
    return {i, j};
}

double distance(const cv::Point2f& point1_line,
                const cv::Point2f& point2_line,
                const cv::Point2f& distant_point) {
    auto [x1, y1] = point1_line;
    auto [x2, y2] = point2_line;
    auto [x0, y0] = distant_point;

    double dx = x2 - x1;
    double dy = y2 - y1;
    double denom = std::sqrt(dx * dx + dy * dy);

    if (denom < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }

    return std::abs(dx * (y1 - y0) - (x1 - x0) * dy) / denom;
}

std::vector<cv::Point2f> getInliers(cv::Point2f line_point_1, cv::Point2f line_point_2, double inlierThreshold, const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> inliers;
    for (auto& point : points) {
        if (distance(line_point_1, line_point_2, point) < inlierThreshold) {
            inliers.push_back(point);
        }
    }
    return inliers;
}

cv::Point2d linearRegression(const std::vector<cv::Point2f>& points) {
    size_t n = points.size();
    if (n < 2) {
        throw std::runtime_error("Not enough points for linear regression");
    }

    double S_x = 0, S_y = 0, S_xy = 0, S_xx = 0;
    for (size_t i = 0; i < n; ++i) {
        auto [x, y] = points[i];
        S_x += x;
        S_y += y;
        S_xy += x * y;
        S_xx += x * x;
    }

    double d = n * S_xx - S_x * S_x;
    if (std::abs(d) < 1e-12) {
        throw std::runtime_error("Degenerate regression");
    }

    double a = (n * S_xy - S_x * S_y) / d;
    double b = (S_y - a * S_x) / n;

    return {a, b};
}

cv::Point2d RANSACforLinearRegression(const std::vector<cv::Point2f>& points, double inlierThreshold, size_t maxIterations) {
    size_t n = points.size();

    if (points.size() < 2) {
        throw std::runtime_error("Not enough points for RANSAC");
    }

    std::vector<std::pair<size_t, std::pair<size_t, size_t>>> lines; // <score, <point_idx_1, point_idx_2>>

    for (size_t i = 0; i < maxIterations; ++i) {
        auto [point_idx_1, point_idx_2] = get2RandomIndex(n);
        cv::Point2f line_point_1 = points[point_idx_1];
        cv::Point2f line_point_2 = points[point_idx_2];


        auto inliers = getInliers(line_point_1, line_point_2, inlierThreshold, points);
        lines.push_back({inliers.size(), {point_idx_1, point_idx_2}});
    }

    auto [best_score, best_line] = *std::max_element(lines.begin(), lines.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });


    auto [point_idx_1, point_idx_2] = best_line;
    std::println("best_score: {}", best_score);
    std::println("best_line: ({}, {})", point_idx_1, point_idx_2);

    auto usable_points = getInliers(points[point_idx_1], points[point_idx_2], inlierThreshold, points);
    return linearRegression(usable_points);
}

std::vector<cv::Point2f> standardPoints2cvPoints(const std::vector<std::pair<double, double>>& points) {
    std::vector<cv::Point2f> cv_points;
    for (const auto& [x, y] : points) {
        cv_points.emplace_back(x, y);
    }
    return cv_points;
}

std::vector<std::pair<double, double>> cvPoints2standardPoints(const std::vector<cv::Point2f>& points) {
    std::vector<std::pair<double, double>>  standard_points;
    for (const auto& point : points) {
        standard_points.emplace_back(point.x, point.y);
    }
    return standard_points;
}

// ----------------------------------------------------- task3 ----------------------

std::tuple<size_t, size_t, size_t, size_t> get4RandomIndex(size_t n) {
    if (n < 4) {
        throw std::invalid_argument("n must be >= 4");
    }

    static std::mt19937 rng(123456);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,...,n-1

    std::shuffle(indices.begin(), indices.end(), rng);

    return {indices[0], indices[1], indices[2], indices[3]};
}

cv::Affine3d PnPbyIdx(
    const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_NED_X,
    const std::vector<size_t>& indices,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs
) {
    if (pixels_x.size() != points_NED_X.size()) {
        throw std::invalid_argument("pixels_x and points_NED_X must have the same size");
    }

    if (indices.size() < 4) {
        throw std::invalid_argument("indices must contain at least 4 elements");
    }

    std::vector<cv::Point2d> curr_pixels_x;
    std::vector<cv::Point3d> curr_points_NED_X;

    curr_pixels_x.reserve(indices.size());
    curr_points_NED_X.reserve(indices.size());

    for (int idx : indices) {
        if (idx < 0 || idx >= static_cast<int>(pixels_x.size())) {
            throw std::out_of_range("index out of range");
        }

        curr_pixels_x.push_back(pixels_x[idx]);
        curr_points_NED_X.push_back(points_NED_X[idx]);
    }

    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(
        curr_points_NED_X,
        curr_pixels_x,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        false,
        indices.size() == 4 ? cv::SOLVEPNP_P3P : cv::SOLVEPNP_ITERATIVE
    );

    if (!ok) {
        // throw std::runtime_error("solvePnP failed");
        return(cv::Matx33d::eye, cv::Vec3d(0, 0, 0));
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    return cv::Affine3d(R, tvec);
}

std::vector<size_t> getPnPInliersIdx(
    const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_X,
    double threshold_in_px,
    const cv::Matx33d& R,
    const cv::Vec3d& t,
    const cv::Matx33d& cameraMatrix
) {
    std::vector<size_t> inliers_idx;

    for (size_t i = 0; i < pixels_x.size(); ++i) {
        cv::Vec3d X(points_X[i].x, points_X[i].y, points_X[i].z);

        cv::Vec3d Xc = R * X + t;

        if (Xc[2] <= 0) {
            continue;
        }

        double x_norm = Xc[0] / Xc[2];
        double y_norm = Xc[1] / Xc[2];

        double u = cameraMatrix(0, 0) * x_norm + cameraMatrix(0, 2);
        double v = cameraMatrix(1, 1) * y_norm + cameraMatrix(1, 2);

        double err = std::sqrt(
            (u - pixels_x[i].x) * (u - pixels_x[i].x) +
            (v - pixels_x[i].y) * (v - pixels_x[i].y)
        );

        if (err < threshold_in_px) {
            inliers_idx.push_back(i);
        }
    }

    return inliers_idx;
}

template<typename T>
std::vector<T> selectByIdx(std::vector<T> inp, std::vector<int> idxs) {
    std::vector<T> out;
    out.reserve(idxs.size());
    for (auto idx : idxs) {
        out.push_back(inp[idx]);
    }
    return out;
}

cv::Affine3d RANSACforPnP(const std::vector<cv::Point2d>& pixels_x,
    const std::vector<cv::Point3d>& points_X,
    double threshold_in_px,
    int max_iterations,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs) {

    if (pixels_x.size() != points_X.size()) throw std::invalid_argument("pixels_x and points_NED_X must have the same size");
    if (pixels_x.size() < 4) throw std::invalid_argument("pixels_x must have at least 4 points");
    if (points_X.size() < 4) throw std::invalid_argument("points_NED_X must have at least 4 points");
    if (threshold_in_px < 0) throw std::invalid_argument("threshold_in_px must be non-negative");

    size_t n = pixels_x.size();

    std::pair<size_t, std::tuple<size_t, size_t, size_t, size_t>> best_model = {0, {1,2,3,4}};

    for (size_t i = 0; i < max_iterations; ++i) {
        auto [point_idx_1, point_idx_2, point_idx_3, point_idx_4] = get4RandomIndex(n);

        cv::Affine3d affine = PnPbyIdx(pixels_x, points_X, {point_idx_1, point_idx_2, point_idx_3, point_idx_4}, cameraMatrix, distCoeffs);
        auto R = affine.rotation();
        auto t = affine.translation();


        size_t inliers_count = getPnPInliersIdx(pixels_x, points_X, threshold_in_px, R, t, cameraMatrix).size();

        if (inliers_count > best_model.first) {
            best_model = {inliers_count, {point_idx_1, point_idx_2, point_idx_3, point_idx_4}};
        }

    }
    auto [best_idx_1, best_idx_2, best_idx_3, best_idx_4] = std::get<1>(best_model);
    cv::Affine3d best_affine = PnPbyIdx(pixels_x, points_X, {best_idx_1, best_idx_2, best_idx_3, best_idx_4}, cameraMatrix, distCoeffs);
    auto best_R = best_affine.rotation();
    auto best_t = best_affine.translation();

    std::vector<size_t> inliers_idx = getPnPInliersIdx(pixels_x, points_X, threshold_in_px, best_R, best_t, cameraMatrix);

    return PnPbyIdx(pixels_x, points_X, inliers_idx, cameraMatrix, distCoeffs);


}


// ------------------------------- task 4 ----------------
