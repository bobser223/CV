//
// Created by Volodymyr Avvakumov on 18.03.2026.
//
#include <opencv2/calib3d.hpp>

#include "utils002.h"


int main() {

    cv::Matx33d R1 = cv::Matx33d::eye();
    cv::Vec3d t1(0, 0, 0);
    cv::Affine3d P_1(R1, t1);

    cv::Vec3d rvec_2(0.1, -0.2, 0.3), t(1.1, -2.1, 3.);
    cv::Affine3d P_2(rvec_2, t);

    double fx = 300, fy = 300, cx = 320, cy = 320;
    cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

    std::vector<cv::Vec3d> X_world_points = {
        {3, 2, 5},
        {10, -3, 5.5},
        {-3, 1, 4.5},
        {8, 2, 4},
        {-2, -3, 4},
        {8, 3, 10},
        {-5, 4, 6},
        {2, -6, 7}
    };

    std::vector<cv::Vec2d> x_pixel_pos_1 = world2Pixels(X_world_points, cameraMatrix, P_1);
    std::vector<cv::Vec2d> x_pixel_pos_2 = world2Pixels(X_world_points, cameraMatrix, P_2);

    cv::Mat E_cv = cv::findEssentialMat(x_pixel_pos_1, x_pixel_pos_2, cameraMatrix);

    auto x_pixel_pos_1_normalized = normalizePixels(x_pixel_pos_1, cameraMatrix);
    auto x_pixel_pos_2_normalized = normalizePixels(x_pixel_pos_2, cameraMatrix);
    cv::Affine3d E_my_8_p_algo = eightPointAlgorithm(x_pixel_pos_1_normalized,x_pixel_pos_2_normalized);
    // std::cout << "E =\n" << E << "\n";

    cv::Mat R_from_cv_function, t_from_cv_function;
    cv::recoverPose(E_cv, x_pixel_pos_1, x_pixel_pos_2, cameraMatrix, R_from_cv_function, t_from_cv_function);


    std::cout << "real R = \n" << cv::Mat(P_2.rotation()) << "\n";
    std::cout << "R from cv = \n" << R_from_cv_function << "\n";
    std::cout << "R from cv but gram = \n" << gramSchmidt(cv::Mat(P_2.rotation())) << "\n";
    std::cout << "R from my 8p algo = \n" << cv::Mat(E_my_8_p_algo.rotation()) << "\n";
    std::cout << "t = " << P_2.translation() << "\n";
    std::cout << "t from cv = " << t_from_cv_function << "\n";
    std::cout << "t from my 8p algo = " << E_my_8_p_algo.translation() << "\n";

}

