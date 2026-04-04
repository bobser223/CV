#include <iostream>


#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>

#include <ceres/ceres.h>

#include "utils003.h"
#include "regressionDrawer.h"





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
            new ceres::HuberLoss(5.0),
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

int main() {
    // testTask1();
    testTask2();
}