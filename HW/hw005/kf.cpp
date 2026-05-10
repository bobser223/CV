//
// Created by Volodymyr Avvakumov on 10.05.2026.
//
#include "kf.h"

#include <iostream>
#include <random>
#include <Eigen/Dense>

using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat36 = Eigen::Matrix<double, 3, 6>;
using Mat63 = Eigen::Matrix<double, 6, 3>;

std::vector<std::tuple<double, Vec3, Vec3>> kalmanFilterAccelerometerGPS(
    double sigma_s,
    double sigma_v,
    double sigma_a,
    double sigma_g,
    std::vector<std::pair<double, Vec3>> acc_motion,
    std::vector<std::pair<double, Vec3>> gps_motion
) {
    std::vector<std::tuple<double, Vec3, Vec3>> filtered_motion;

    Vec6 x_state = Vec6::Zero(); // s_x, s_y, s_z, v_x, v_y, v_z

    x_state.segment<3>(0) = gps_motion.front().second; // first position from GPS
    std::size_t gps_index = 1;

    Mat6 S_aka_P_aka_covariance = Mat6::Zero();
    S_aka_P_aka_covariance(0, 0) = sigma_s * sigma_s;
    S_aka_P_aka_covariance(1, 1) = sigma_s * sigma_s;
    S_aka_P_aka_covariance(2, 2) = sigma_s * sigma_s;

    S_aka_P_aka_covariance(3, 3) = sigma_v * sigma_v;
    S_aka_P_aka_covariance(4, 4) = sigma_v * sigma_v;
    S_aka_P_aka_covariance(5, 5) = sigma_v * sigma_v;


    // GPS measurement matrix: z = Cx
    Mat36 C = Mat36::Zero();
    C.block<3, 3>(0, 0) = Mat3::Identity();

    // GPS noise covariance
    Mat3 R = sigma_g * sigma_g * Mat3::Identity();




    double dt = 0.0;
    double previous_timestamp = acc_motion.front().first;
    for (const auto&[timestamp, a]: acc_motion) {
        dt = timestamp - previous_timestamp;
        previous_timestamp = timestamp;

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
        Mat6 Q = Mat6::Zero();
        Q.block<3, 3>(3, 3) = sigma_a * sigma_a * Mat3::Identity();

        // x_k = A x_{k-1} + B a_k
        x_state = A * x_state + B * a;

        // S_k = A S_{k-1} A^T + Q

        x_state = A * x_state + B * a;
        S_aka_P_aka_covariance =
            A * S_aka_P_aka_covariance * A.transpose() + Q;

        while (
            gps_index < gps_motion.size() &&
            gps_motion[gps_index].first <= timestamp
        ) {
            const Vec3 z = gps_motion[gps_index].second;

            const Vec3 innovation = z - C * x_state;

            const Mat3 innovation_covariance =
                C * S_aka_P_aka_covariance * C.transpose() + R;

            const Mat63 K =
                S_aka_P_aka_covariance *
                C.transpose() *
                innovation_covariance.inverse();

            x_state = x_state + K * innovation;

            S_aka_P_aka_covariance =
                S_aka_P_aka_covariance - K * C * S_aka_P_aka_covariance;

            ++gps_index;
        }

        std::cout << "t = " << timestamp
                  << " position = " << x_state.segment<3>(0).transpose()
                  << " velocity = " << x_state.segment<3>(3).transpose()
                  << "\n";

        filtered_motion.push_back(std::make_tuple(acc_motion.front().first, x_state.segment<3>(0), x_state.segment<3>(3)));
    }



    return filtered_motion;
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


struct SimulatedMotion {
    std::vector<std::pair<double, Vec3>> acc_motion;
    std::vector<std::pair<double, Vec3>> gps_motion;
    std::vector<std::pair<double, Vec3>> true_positions;
};

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

    // =========================
    // Accelerometer 100Hz
    // =========================

    for (double t = 0.0; t <= total_time; t += acc_dt) {
        Vec3 a_true = trueAcceleration(t);

        Vec3 a_measured{
            a_true.x() + acc_noise(rng),
            a_true.y() + acc_noise(rng),
            a_true.z() + acc_noise(rng)
        };

        result.acc_motion.push_back({t, a_measured});

        Vec3 s_true = truePosition(t);
        result.true_positions.push_back({t, s_true});
    }

    // =========================
    // GPS, for example 1Hz
    // =========================

    for (double t = 0.0; t <= total_time; t += gps_dt) {
        Vec3 s_true = truePosition(t);

        Vec3 gps_measured{
            s_true.x() + gps_noise(rng),
            s_true.y() + gps_noise(rng),
            s_true.z() + gps_noise(rng)
        };

        result.gps_motion.push_back({t, gps_measured});
    }

    return result;
}


void printTrajectoryError(
    const std::vector<std::pair<double, Vec3>>& estimated_positions,
    const std::vector<std::pair<double, Vec3>>& true_positions
) {
    double sum_error = 0.0;
    double max_error = 0.0;

    const std::size_t n = std::min(
        estimated_positions.size(),
        true_positions.size()
    );

    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 estimated = estimated_positions[i].second;
        const Vec3 truth = true_positions[i].second;

        const double error = (estimated - truth).norm();

        sum_error += error;
        max_error = std::max(max_error, error);
    }

    const double mean_error = sum_error / static_cast<double>(n);

    std::cout << "Mean position error = " << mean_error << "\n";
    std::cout << "Max position error  = " << max_error << "\n";
}

