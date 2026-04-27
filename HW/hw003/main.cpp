#include <iostream>


#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "utils003.h"
#include "regressionDrawer.h"
#include "task4.h"


struct SyntheticPnPData {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat rvec_gt;
    cv::Mat tvec_gt;
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> pixels2d;
    std::vector<int> inlier_mask_gt;
};

SyntheticPnPData loadSyntheticPnP(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    SyntheticPnPData data;
    cv::Mat points3d_mat, pixels2d_mat, inlier_mask_mat;

    fs["cameraMatrix"] >> data.cameraMatrix;
    fs["distCoeffs"] >> data.distCoeffs;
    fs["rvec_gt"] >> data.rvec_gt;
    fs["tvec_gt"] >> data.tvec_gt;
    fs["points3d"] >> points3d_mat;
    fs["pixels2d"] >> pixels2d_mat;
    fs["inlier_mask_gt"] >> inlier_mask_mat;

    if (points3d_mat.cols != 3) {
        throw std::runtime_error("points3d must be Nx3");
    }
    if (pixels2d_mat.cols != 2) {
        throw std::runtime_error("pixels2d must be Nx2");
    }

    data.points3d.reserve(points3d_mat.rows);
    for (int i = 0; i < points3d_mat.rows; ++i) {
        data.points3d.emplace_back(
            points3d_mat.at<double>(i, 0),
            points3d_mat.at<double>(i, 1),
            points3d_mat.at<double>(i, 2)
        );
    }

    data.pixels2d.reserve(pixels2d_mat.rows);
    for (int i = 0; i < pixels2d_mat.rows; ++i) {
        data.pixels2d.emplace_back(
            pixels2d_mat.at<double>(i, 0),
            pixels2d_mat.at<double>(i, 1)
        );
    }

    data.inlier_mask_gt.reserve(inlier_mask_mat.rows);
    for (int i = 0; i < inlier_mask_mat.rows; ++i) {
        data.inlier_mask_gt.push_back((int)inlier_mask_mat.at<uchar>(i, 0));
    }

    return data;
}

double rotationErrorDeg(const cv::Matx33d& R_est, const cv::Matx33d& R_gt) {
    cv::Matx33d dR = R_est * R_gt.t();
    double c = (cv::trace(dR) - 1.0) * 0.5;
    c = std::clamp(c, -1.0, 1.0);
    return std::acos(c) * 180.0 / CV_PI;
}

double translationError(const cv::Vec3d& t_est, const cv::Vec3d& t_gt) {
    return cv::norm(t_est - t_gt);
}


auto testTask1() {
    const std::string filename = PATH_TO_DATA + "//hw003//points.txt";

    double inlierThreshold = 1.0;

    size_t maxIterations = 1000;

    std::vector<cv::Point2f> points = readPointsFromFile(filename);

    cv::Point2d lr = linearRegression(points);
    cv::Point2d ransac = RANSACforLinearRegression(points, inlierThreshold, maxIterations);

    drawRegressionResult(points, lr,
                             "Ordinary Linear Regression",
                             "linear_regression.png");

    drawRegressionResult(points, ransac,
                             "RANSAC Linear Regression",
                             "ransac_regression.png");

    std::cout << "Saved: linear_regression.png\n";
    std::cout << "Saved: ransac_regression.png\n";
    std::cout << "Linear regression: a = " << lr.x << ", b = " << lr.y << '\n';
    std::cout << "RANSAC regression: a = " << ransac.x << ", b = " << ransac.y << '\n';

}

struct LinearRegression{
    LinearRegression(const double x_, const double y_): x(x_), y(y_) {}
    ~LinearRegression() = default;

    template <typename T>
    bool operator()(const T* const coef, T* residuals) const {
        const T& a = coef[0];
        const T& b = coef[1];
        residuals[0] = a * x + b - y;
        return true;
    }

private:
    double x = 0.0;
    double y = 0.0;
};

auto testTask2() {
    const std::string filename = PATH_TO_DATA + "//hw003//falseRegressionPoints.txt";

    auto points = cvPoints2standardPoints(readPointsFromFile(filename));

    // std::vector<std::pair<double, double>> points = {
    //     {0.0, 1.0},
    //     {1.0, 3.0},
    //     {2.0, 5.1},
    //     {3.0, 7.2},
    //     {4.0, 9.1}
    // };


    double params[2] = {0.0, 0.0}; // a, b
    ceres::Problem problem;

    for (const auto& [x, y] : points) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LinearRegression, 1, 2>(
                new LinearRegression(x, y)
            ),
            nullptr,
            params
        );
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "a = " << params[0] << "\n";
    std::cout << "b = " << params[1] << "\n";

    cv::Point2d params_cv = {params[0], params[1]};

    std::vector<cv::Point2f> points_cv = standardPoints2cvPoints(points);

    drawRegressionResult(points_cv, params_cv,
                             "Ordinary Linear Regression",
                             "linear_regression_by_ceres.png");

//------------------------ loss --------
    double params_loss[2] = {0.0, 0.0}; // a, b

    ceres::Problem problem_loss;
    for (const auto& [x, y] : points) {
        problem_loss.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LinearRegression, 1, 2>(
                new LinearRegression(x, y)
            ),
            new ceres::HuberLoss(0.1),
            params_loss
        );
    }
    ceres::Solver::Options options_loss;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary_loss;
    ceres::Solve(options_loss, &problem_loss, &summary_loss);

    std::cout << "a = " << params_loss[0] << "\n";
    std::cout << "b = " << params_loss[1] << "\n";

    cv::Point2d params_cv_loss = {params_loss[0], params_loss[1]};

    std::vector<cv::Point2f> points_cv_loss = standardPoints2cvPoints(points);

    drawRegressionResult(points_cv_loss, params_cv_loss,
                             "Loss Linear Regression",
                             "linear_regression_by_ceres_loss.png");

    return 0;
}


void task3SubtestByTestNuber(const std::string& number) {
    SyntheticPnPData data = loadSyntheticPnP(PATH_TO_DATA + "hw003//synthetic_pnp_"+ number +".yaml");
    cv::Affine3d est = RANSACforPnP(
        data.pixels2d,
        data.points3d,
        3.0,          // threshold_in_px
        1000,         // max_iterations
        data.cameraMatrix,
        data.distCoeffs
    );

    cv::Mat R_gt_mat;
    cv::Rodrigues(data.rvec_gt, R_gt_mat);

    cv::Matx33d R_gt = R_gt_mat;
    cv::Vec3d t_gt(
        data.tvec_gt.at<double>(0,0),
        data.tvec_gt.at<double>(1,0),
        data.tvec_gt.at<double>(2,0)
    );

    cv::Matx33d R_est = est.rotation();
    cv::Vec3d t_est = est.translation();

    std::cout << "rotation error [deg]: " << rotationErrorDeg(R_est, R_gt) << "\n";
    std::cout << "translation error   : " << translationError(t_est, t_gt) << "\n";

    auto inliers = getPnPInliersIdx(
        data.pixels2d,
        data.points3d,
        3.0,
        R_est,
        t_est,
        data.cameraMatrix
    );

    std::cout << "estimated inliers: " << inliers.size() << "\n";
}

void testTask3() {
    std::println("Test01: outlier_ratio=0.0, noise_std_px=0.1");
    task3SubtestByTestNuber("1");
    std::println("Test02: outlier_ratio=0.0, noise_std_px= 1.0");
    task3SubtestByTestNuber("2");
    std::println("Test03: outlier_ratio=0.2, noise_std_px= 1.5");
    task3SubtestByTestNuber("3");
    std::println("Test04: outlier_ratio=0.4, noise_std_px= 2.0");
    task3SubtestByTestNuber("4");
    std::println("Test05: outlier_ratio=0.5, noise_std_px= 3.0");
}

void testTask4() {
    const std::string base = PATH_TO_DATA + "//hw003//";

    const std::vector<std::string> filenames = {
        "sample_linear_dominant.txt",
        "sample_quadratic_dominant.txt",
        "sample_exp_dominant.txt"
    };

    const double inlierThreshold = 2.0;
    const size_t maxIterations = 1000;

    for (const auto& name : filenames) {
        std::vector<cv::Point2d> points =
            points2f2points2d(readPointsFromFile(base + name));

        Model winner = modelSelection(points, inlierThreshold, maxIterations);

        std::cout << "==============================\n";
        std::cout << "file: " << name << "\n";
        std::cout << "winner: " << winner.name << "\n";
        std::cout << "inliers: " << winner.inliers_count << "\n";
        std::cout << "coefficients: ";

        for (double c : winner.coefficients) {
            std::cout << c << " ";
        }

        std::cout << "\n";
        std::cout << "==============================\n";
    }
}

int main() {
    // testTask1();
    // testTask2();

    // testTask3();
    testTask4();
}