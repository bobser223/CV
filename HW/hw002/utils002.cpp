//
// Created by Volodymyr Avvakumov on 17.03.2026.
//

#include "utils002.h"


cv::Mat eight_point_algorithm(const std::array<cv::Vec3d, 8>& x_0_points, const std::array<cv::Vec3d, 8>& x_points) {
    cv::Mat matrixToSolve(8, 9, CV_64FC1);
    cv::Mat essentialMatrixCoefficients(1, 8, CV_64FC1);

    for (int i = 0; i < 8; ++i) {
        double u0 = x_0_points[i][0];
        double v0 = x_0_points[i][1];
        double u = x_points[i][0];
        double v = x_points[i][1];

        // (u*u_0,u*v_0,u,v*u_0,v*v_0,v,u_0,v_0, 1)
        matrixToSolve.at<double>(i, 0) = u*u0;
        matrixToSolve.at<double>(i, 1) = u*v0;
        matrixToSolve.at<double>(i, 2) = u;
        matrixToSolve.at<double>(i, 3) = v*u0;
        matrixToSolve.at<double>(i, 4) = v*v0;
        matrixToSolve.at<double>(i, 5) = v;
        matrixToSolve.at<double>(i, 6) = u0;
        matrixToSolve.at<double>(i, 7) = v0;
        matrixToSolve.at<double>(i, 8) = 1;
    }

    cv::Mat w, U, Vt;
    cv::SVD::compute(matrixToSolve, w, U, Vt, cv::SVD::FULL_UV);

    essentialMatrixCoefficients = Vt.row(8).clone();
    cv::Mat essentialMatrix = essentialMatrixCoefficients.reshape(1, 3).clone();;







    return cv::Mat();
}
