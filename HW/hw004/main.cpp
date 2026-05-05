//
// Created by Volodymyr Avvakumov on 20.04.2026.
//
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "utils002.h"
#include "cv_types.h"


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

struct QuadraticFunction //f =  (3x-y+1)^2 + (4z+x)^2 + (z -5y + 10x)^2
{
    template <typename T>
    bool operator()(const T* const coef, T* residuals) const {
        const T& x = coef[0];
        const T& y = coef[1];
        const T& z = coef[2];

        residuals[0] = static_cast<T>(3)*x -y + static_cast<T>(1);
        residuals[1] = static_cast<T>(4)*z+x;
        residuals[2] = z - static_cast<T>(5)*y + static_cast<T>(10)*x;
        return true;
    }

};



void task001() {
    double coef[3] = {0.0, 0.0, 0.0}; // x, y, z

    ceres::Problem problem;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<QuadraticFunction, 3, 3>(
            new QuadraticFunction()
        ),
        nullptr,
        coef
    );

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";

    std::cout << "Found minimum at:\n";
    std::cout << "x = " << coef[0] << "\n";
    std::cout << "y = " << coef[1] << "\n";
    std::cout << "z = " << coef[2] << "\n";

    double r1 = 3.0 * coef[0] - coef[1] + 1.0;
    double r2 = coef[0] + 4.0 * coef[2];
    double r3 = coef[2] - 5.0 * coef[1] + 10.0 * coef[0];

    double f = r1 * r1 + r2 * r2 + r3 * r3;

    std::cout << "f(x,y,z) = " << f << "\n";
}


void task002Bundle() {
    std::vector<Eigen::Vector2d> projection_1 = {
        { 0.0,     0.0},
        { 0.2,     0.0},
        {-0.2,     0.1},
        { 0.125,  -0.125},
        {-0.05,   -0.133333333333}
    };

    std::vector<Eigen::Vector2d> projection_2 = {
        {-0.125,           0.0},
        { 0.1,             0.0},
        {-0.3,             0.1},
        { 0.0,            -0.125},
        {-0.133333333333, -0.133333333333}
    };

    double q1[] = {0.0, 0.0, 0.0, 1.0};
    double q2[] = {q1[0], q1[1], q1[2], q1[3]};
    double t1[] = {0.0, 0.0, 0.0};
    double t2[] = {0.0, 0.0, 0.0};
    std::vector<Eigen::Vector3d> X_est = {
        { 0.1,  -0.1, 3.5},
        { 0.8,   0.1, 4.5},
        {-0.8,   0.3, 4.5},
        { 0.7,  -0.3, 3.5},
        {-0.1,  -0.6, 5.5}
    };


    ceres::Problem problem;

    for (int i = 0; i <  X_est.size(); ++i ) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BundleAdjustment, 2, 4, 3, 3>(
                new BundleAdjustment(projection_1[i])
            ),
            nullptr,
            q1,
            t1,
            X_est[i].data()
        );
    }

    for (int i = 0; i <  X_est.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BundleAdjustment, 2, 4, 3, 3>(
                new BundleAdjustment(projection_2[i])
            ),
            nullptr,
            q2,
            t2,
            X_est[i].data()
        );
    }

    problem.SetParameterBlockConstant(q1);
    problem.SetParameterBlockConstant(t1);
    // problem.SetParameterBlockConstant(q2);
    // problem.SetParameterBlockConstant(t2);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";

    for (size_t i = 0; i < X_est.size(); ++i) {
        std::cout << "X[" << i << "] = "
                  << X_est[i].transpose()
                  << std::endl;
    }




}

void task002() {
    std::vector<vec2> projection_pixels_1 = {
        { 0.0,     0.0},
        { 0.2,     0.0},
        {-0.2,     0.1},
        { 0.125,  -0.125},
        {-0.05,   -0.133333333333}
    };

    std::vector<vec2> projection_pixels_2 = {
        {-0.125,           0.0},
        { 0.1,             0.0},
        {-0.3,             0.1},
        { 0.0,            -0.125},
        {-0.133333333333, -0.133333333333}
    };



    cv::Matx33d cameraMatrix(
        800.0,   0.0, 320.0,
          0.0, 800.0, 240.0,
          0.0,   0.0,   1.0
    );



    auto [R, t, points_NED] = estimateSLAM(projection_pixels_1, projection_pixels_2, cameraMatrix);


}

void testEstimateSLAM() {
    cv::Matx33d cameraMatrix(
        800.0,   0.0, 320.0,
          0.0, 800.0, 240.0,
          0.0,   0.0,   1.0
    );


    std::vector<point3> points_true = {
        { 0.0,  0.0, 5.0},
        { 1.0,  0.2, 6.0},
        {-1.0,  0.5, 5.5},
        { 0.5, -0.7, 4.5},
        {-0.4, -0.6, 7.0},
        { 1.2, -0.4, 8.0},
        {-1.4,  0.8, 6.5},
        { 0.3,  1.0, 7.5}
    };


    cv::Matx33d R1 = cv::Matx33d::eye();
    vec3 t1(0.0, 0.0, 0.0);
    cv::Affine3d affine1(R1, t1);


    cv::Vec3d rvec_true(0.03, -0.08, 0.02);

    cv::Mat R_true_mat;
    cv::Rodrigues(rvec_true, R_true_mat);

    cv::Matx33d R_true(
        R_true_mat.at<double>(0, 0), R_true_mat.at<double>(0, 1), R_true_mat.at<double>(0, 2),
        R_true_mat.at<double>(1, 0), R_true_mat.at<double>(1, 1), R_true_mat.at<double>(1, 2),
        R_true_mat.at<double>(2, 0), R_true_mat.at<double>(2, 1), R_true_mat.at<double>(2, 2)
    );


    vec3 t_true(-1.0, 0.0, 0.0);

    cv::Affine3d affine2(R_true, t_true);

    std::vector<vec2> projection_pixels_1 =
        projectPoints(points_true, affine1, cameraMatrix);

    std::vector<vec2> projection_pixels_2 =
        projectPoints(points_true, affine2, cameraMatrix);

    std::cout << "Generated projections:\n";
    for (size_t i = 0; i < projection_pixels_1.size(); ++i) {
        std::cout << "point " << i << "\n";
        std::cout << "  img1 = " << projection_pixels_1[i] << "\n";
        std::cout << "  img2 = " << projection_pixels_2[i] << "\n";
    }

    auto [R_est, t_est, points_est] =
        estimateSLAM(projection_pixels_1, projection_pixels_2, cameraMatrix);

    std::cout << "\n===== TRUE R =====\n";
    std::cout << cv::Mat(R_true) << "\n";

    std::cout << "\n===== ESTIMATED R =====\n";
    std::cout << cv::Mat(R_est) << "\n";

    std::cout << "\n===== TRUE t =====\n";
    std::cout << t_true << "\n";

    std::cout << "\n===== ESTIMATED t =====\n";
    std::cout << t_est << "\n";

    std::cout << "\n===== TRUE POINTS vs ESTIMATED POINTS =====\n";
    for (size_t i = 0; i < points_true.size() && i < points_est.size(); ++i) {
        std::cout << "point " << i << "\n";
        std::cout << "  true = "
                  << points_true[i].x << " "
                  << points_true[i].y << " "
                  << points_true[i].z << "\n";

        std::cout << "  est  = "
                  << points_est[i].x << " "
                  << points_est[i].y << " "
                  << points_est[i].z << "\n";
    }
}

void task003() {
    testBinocularSLAMWithNoise();
}


int main() {
    // task001();
    // task002Bundle();
    // task002();
    // testEstimateSLAM();
    // task003();
    task004();
}