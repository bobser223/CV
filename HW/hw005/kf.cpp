//
// Created by Volodymyr Avvakumov on 10.05.2026.
//

#include "kf.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <string>

void plotTrajectory2D(
    const FilteredMotion& filtered_motion,
    const std::vector<std::pair<double, Vec3>>& true_positions,
    const std::string& window_name,
    const std::string& save_path
) {
    const std::size_t n = std::min(filtered_motion.size(), true_positions.size());

    if (n == 0) {
        std::cout << "No trajectory points to plot.\n";
        return;
    }

    const int width = 1200;
    const int height = 800;
    const int margin = 80;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();

    // знайти межі по true + estimated
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 true_p = true_positions[i].second;
        const auto& [timestamp, estimated_p, estimated_v] = filtered_motion[i];

        min_x = std::min(min_x, true_p.x());
        max_x = std::max(max_x, true_p.x());
        min_y = std::min(min_y, true_p.y());
        max_y = std::max(max_y, true_p.y());

        min_x = std::min(min_x, estimated_p.x());
        max_x = std::max(max_x, estimated_p.x());
        min_y = std::min(min_y, estimated_p.y());
        max_y = std::max(max_y, estimated_p.y());
    }

    // padding
    const double dx = std::max(1e-9, max_x - min_x);
    const double dy = std::max(1e-9, max_y - min_y);

    min_x -= 0.05 * dx;
    max_x += 0.05 * dx;
    min_y -= 0.05 * dy;
    max_y += 0.05 * dy;

    auto toPixel = [&](double x, double y) -> cv::Point {
        const double px =
            margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin);

        const double py =
            height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin);

        return cv::Point(
            static_cast<int>(std::round(px)),
            static_cast<int>(std::round(py))
        );
    };

    // осі/рамка
    cv::rectangle(
        canvas,
        cv::Point(margin, margin),
        cv::Point(width - margin, height - margin),
        cv::Scalar(200, 200, 200),
        1
    );

    // підписи меж
    cv::putText(
        canvas,
        "x from " + std::to_string(min_x) + " to " + std::to_string(max_x),
        cv::Point(margin, height - 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );

    cv::putText(
        canvas,
        "y from " + std::to_string(min_y) + " to " + std::to_string(max_y),
        cv::Point(margin, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );

    // true trajectory: зелена
    for (std::size_t i = 1; i < n; ++i) {
        const Vec3 p0 = true_positions[i - 1].second;
        const Vec3 p1 = true_positions[i].second;

        cv::line(
            canvas,
            toPixel(p0.x(), p0.y()),
            toPixel(p1.x(), p1.y()),
            cv::Scalar(0, 180, 0),
            2,
            cv::LINE_AA
        );
    }

    // estimated trajectory: червона
    for (std::size_t i = 1; i < n; ++i) {
        const auto& [t0, est0, vel0] = filtered_motion[i - 1];
        const auto& [t1, est1, vel1] = filtered_motion[i];

        cv::line(
            canvas,
            toPixel(est0.x(), est0.y()),
            toPixel(est1.x(), est1.y()),
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA
        );
    }

    // стартові точки
    {
        const Vec3 true_start = true_positions.front().second;
        const auto& [t0, est_start, vel0] = filtered_motion.front();

        cv::circle(canvas, toPixel(true_start.x(), true_start.y()), 6, cv::Scalar(0, 180, 0), -1);
        cv::circle(canvas, toPixel(est_start.x(), est_start.y()), 6, cv::Scalar(0, 0, 255), -1);

        cv::putText(
            canvas,
            "start",
            toPixel(true_start.x(), true_start.y()) + cv::Point(10, -10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA
        );
    }

    // кінцеві точки
    {
        const Vec3 true_end = true_positions.back().second;
        const auto& [t1, est_end, vel1] = filtered_motion.back();

        cv::circle(canvas, toPixel(true_end.x(), true_end.y()), 6, cv::Scalar(0, 120, 0), 2);
        cv::circle(canvas, toPixel(est_end.x(), est_end.y()), 6, cv::Scalar(0, 0, 180), 2);
    }

    // легенда
    cv::line(canvas, cv::Point(850, 60), cv::Point(900, 60), cv::Scalar(0, 180, 0), 2, cv::LINE_AA);
    cv::putText(canvas, "True trajectory", cv::Point(910, 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    cv::line(canvas, cv::Point(850, 100), cv::Point(900, 100), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    cv::putText(canvas, "Kalman trajectory", cv::Point(910, 105),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    cv::putText(
        canvas,
        "2D trajectory plot (x-y)",
        cv::Point(40, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );

    cv::imwrite(save_path, canvas);
    cv::imshow(window_name, canvas);
    cv::waitKey(0);
}

Vec3 truePosition(double t) {
    return Vec3{
        t,
        std::sin(t),
        0.0
    };
}

Vec3 trueVelocity(double t) {
    return Vec3{
        1.0,
        std::cos(t),
        0.0
    };
}

Vec3 trueAcceleration(double t) {
    return Vec3{
        0.0,
        -std::sin(t),
        0.0
    };
}

SimulatedMotion simulateMotion(
    double total_time,
    double acc_frequency,
    double gps_frequency,
    double sigma_a,
    double sigma_g
) {
    SimulatedMotion result;

    const double acc_dt = 1.0 / acc_frequency;
    const double gps_dt = 1.0 / gps_frequency;

    std::default_random_engine rng(42);

    std::normal_distribution<double> acc_noise(0.0, sigma_a);
    std::normal_distribution<double> gps_noise(0.0, sigma_g);

    // Accelerometer 100 Hz
    for (double t = 0.0; t <= total_time; t += acc_dt) {
        const Vec3 a_true = trueAcceleration(t);

        const Vec3 a_measured{
            a_true.x() + acc_noise(rng),
            a_true.y() + acc_noise(rng),
            a_true.z() + acc_noise(rng)
        };

        result.acc_motion.push_back({t, a_measured});
        result.true_positions.push_back({t, truePosition(t)});
    }

    // GPS 5 Hz
    for (double t = 0.0; t <= total_time; t += gps_dt) {
        const Vec3 s_true = truePosition(t);

        const Vec3 gps_measured{
            s_true.x() + gps_noise(rng),
            s_true.y() + gps_noise(rng),
            s_true.z() + gps_noise(rng)
        };

        result.gps_motion.push_back({t, gps_measured});
    }

    return result;
}

FilteredMotion kalmanFilterAccelerometerGPS(
    double sigma_s,
    double sigma_v,
    double sigma_a,
    double sigma_g,
    const std::vector<std::pair<double, Vec3>>& acc_motion,
    const std::vector<std::pair<double, Vec3>>& gps_motion
) {
    FilteredMotion filtered_motion;

    if (acc_motion.empty()) {
        return filtered_motion;
    }

    Vec6 x_state = Vec6::Zero(); // s_x, s_y, s_z, v_x, v_y, v_z

    std::size_t gps_index = 0;

    // Початкову позицію беремо з першого GPS, якщо він є.
    if (!gps_motion.empty()) {
        x_state.segment<3>(0) = gps_motion.front().second;
        gps_index = 1;
    }

    // S_{0|0}
    Mat6 S_aka_P_aka_covariance = Mat6::Zero();

    S_aka_P_aka_covariance(0, 0) = sigma_s * sigma_s;
    S_aka_P_aka_covariance(1, 1) = sigma_s * sigma_s;
    S_aka_P_aka_covariance(2, 2) = sigma_s * sigma_s;

    S_aka_P_aka_covariance(3, 3) = sigma_v * sigma_v;
    S_aka_P_aka_covariance(4, 4) = sigma_v * sigma_v;
    S_aka_P_aka_covariance(5, 5) = sigma_v * sigma_v;

    // GPS measurement matrix: z = Cx
    // C = [ I  O ]
    Mat36 C = Mat36::Zero();
    C.block<3, 3>(0, 0) = Mat3::Identity();

    // GPS noise covariance
    Mat3 R = sigma_g * sigma_g * Mat3::Identity();

    double previous_timestamp = acc_motion.front().first;

    for (const auto& [timestamp, a] : acc_motion) {
        const double dt = timestamp - previous_timestamp;
        previous_timestamp = timestamp;

        // =========================
        // 1. PREDICTION BY ACCEL
        // =========================

        Mat6 A = Mat6::Identity();
        Mat63 B = Mat63::Zero();

        // A = [ I  dtI ]
        //     [ O   I  ]
        A.block<3, 3>(0, 3) = dt * Mat3::Identity();

        // B = [ O   ]
        //     [ dtI ]
        B.block<3, 3>(3, 0) = dt * Mat3::Identity();

        // Q = [ O        O      ]
        //     [ O  sigma_a^2 I ]
        //
        // Саме так, як у лекції: шум акселерометра додаємо в блок швидкості.
        Mat6 Q = Mat6::Zero();
        Q.block<3, 3>(3, 3) = sigma_a * sigma_a * Mat3::Identity();

        // x_k = A x_{k-1} + B a_k
        x_state = A * x_state + B * a;

        // S_k = A S_{k-1} A^T + Q
        S_aka_P_aka_covariance =
            A * S_aka_P_aka_covariance * A.transpose() + Q;

        // =========================
        // 2. GPS UPDATE IF AVAILABLE
        // =========================

        while (
            gps_index < gps_motion.size() &&
            gps_motion[gps_index].first <= timestamp
        ) {
            const Vec3 z = gps_motion[gps_index].second;

            // z - Cx
            const Vec3 innovation = z - C * x_state;

            // C S C^T + R
            const Mat3 innovation_covariance =
                C * S_aka_P_aka_covariance * C.transpose() + R;

            // K = S C^T (C S C^T + R)^(-1)
            const Mat63 K =
                S_aka_P_aka_covariance *
                C.transpose() *
                innovation_covariance.inverse();

            // x = x + K(z - Cx)
            x_state = x_state + K * innovation;

            // S = S - K C S
            S_aka_P_aka_covariance =
                S_aka_P_aka_covariance - K * C * S_aka_P_aka_covariance;

            ++gps_index;
        }

        const Vec3 estimated_position = x_state.segment<3>(0);
        const Vec3 estimated_velocity = x_state.segment<3>(3);

        filtered_motion.push_back({
            timestamp,
            estimated_position,
            estimated_velocity
        });
    }

    return filtered_motion;
}

void printTrajectoryError(
    const FilteredMotion& filtered_motion,
    const std::vector<std::pair<double, Vec3>>& true_positions
) {
    const std::size_t n = std::min(filtered_motion.size(), true_positions.size());

    if (n == 0) {
        std::cout << "No trajectory points to compare.\n";
        return;
    }

    double sum_error = 0.0;
    double max_error = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const auto& [timestamp, estimated_position, estimated_velocity] = filtered_motion[i];
        const Vec3 true_position = true_positions[i].second;

        const double error = (estimated_position - true_position).norm();

        sum_error += error;
        max_error = std::max(max_error, error);
    }

    const double mean_error = sum_error / static_cast<double>(n);

    std::cout << "Mean position error = " << mean_error << "\n";
    std::cout << "Max position error  = " << max_error << "\n";
}

void testKalmanFilterAccelerometerGPS()
{
    const double total_time = 200.0;

    const double acc_frequency = 100.0; // 100 Hz
    const double gps_frequency = 5.0;   // 5 Hz

    const double sigma_s = 1.0;
    const double sigma_v = 5.0;

    const double sigma_a = 0.05;
    const double sigma_g = 0.5;

    SimulatedMotion motion = simulateMotion(
        total_time,
        acc_frequency,
        gps_frequency,
        sigma_a,
        sigma_g
    );

    FilteredMotion filtered_motion = kalmanFilterAccelerometerGPS(
        sigma_s,
        sigma_v,
        sigma_a,
        sigma_g,
        motion.acc_motion,
        motion.gps_motion
    );

    printTrajectoryError(
        filtered_motion,
        motion.true_positions
    );


    plotTrajectory2D(
    filtered_motion,
    motion.true_positions,
    "True vs Kalman trajectory",
    "trajectory.png");

    for (std::size_t i = 0; i < filtered_motion.size(); i += 100) {
        const auto& [timestamp, estimated_position, estimated_velocity] = filtered_motion[i];
        const Vec3 true_position = motion.true_positions[i].second;

        std::cout
            << "t = " << timestamp
            << " true = " << true_position.transpose()
            << " estimated = " << estimated_position.transpose()
            << " error = " << (estimated_position - true_position).norm()
            << "\n";
    }
}