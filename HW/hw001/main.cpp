#include <iostream>

#include "scane.h"


int main() {
    auto objects = createObjects();
    const int imageWidth = 1280;
    const int imageHeight = 720;


    // camera pose in world (NED)
    cv::Vec3d C_world(0, 0, 0);

    // NED -> camera: x(right)=E, y(down)=D, z(forward)=N
    cv::Matx33d R_cw(
        0, 1, 0,
        0, 0, 1,
        1, 0, 0
    );

    cv::Vec3d t = -(R_cw * C_world);
    cv::Affine3d P(R_cw, t);

    const double fx = 900.0;
    const double fy = 900.0;
    const double cx = imageWidth * 0.5;
    const double cy = imageHeight * 0.5;
    const cv::Matx33d K(
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    );

    cv::Mat img = project2d(P, objects, imageWidth, imageHeight, K);



    cv::imshow("render", img);
    cv::waitKey(0);

    return 0;
}
