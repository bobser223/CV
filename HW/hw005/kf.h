//
// Created by Volodymyr Avvakumov on 10.05.2026.
//

#ifndef CODE_KF_H
#define CODE_KF_H
//
// Created by Volodymyr Avvakumov on 10.05.2026.
//

#ifndef KF_H
#define KF_H

#include <Eigen/Dense>

#include <tuple>
#include <utility>
#include <vector>

using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat36 = Eigen::Matrix<double, 3, 6>;
using Mat63 = Eigen::Matrix<double, 6, 3>;

struct SimulatedMotion {
    std::vector<std::pair<double, Vec3>> acc_motion;
    std::vector<std::pair<double, Vec3>> gps_motion;
    std::vector<std::pair<double, Vec3>> true_positions;
};

using FilteredMotion = std::vector<std::tuple<double, Vec3, Vec3>>;
// tuple: timestamp, estimated_position, estimated_velocity

#include <opencv2/opencv.hpp>
#include <string>

void plotTrajectory2D(
    const FilteredMotion& filtered_motion,
    const std::vector<std::pair<double, Vec3>>& true_positions,
    const std::string& window_name,
    const std::string& save_path
);
Vec3 truePosition(double t);

Vec3 trueVelocity(double t);

Vec3 trueAcceleration(double t);

SimulatedMotion simulateMotion(
    double total_time,
    double acc_frequency,
    double gps_frequency,
    double sigma_a,
    double sigma_g
);

FilteredMotion kalmanFilterAccelerometerGPS(
    double sigma_s,
    double sigma_v,
    double sigma_a,
    double sigma_g,
    const std::vector<std::pair<double, Vec3>>& acc_motion,
    const std::vector<std::pair<double, Vec3>>& gps_motion
);

void printTrajectoryError(
    const FilteredMotion& filtered_motion,
    const std::vector<std::pair<double, Vec3>>& true_positions
);

void testKalmanFilterAccelerometerGPS();

#endif // KF_H




#endif //CODE_KF_H
