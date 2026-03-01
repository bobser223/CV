#include <iostream>

#include "scane.h"


int main() {
    auto objects = createObjects();


    // camera pose in world (NED)
    cv::Vec3d C_world(0, 0, 0);

    // rotation world->camera
    cv::Matx33d R_cw = cv::Matx33d::eye(); // no rotation

    cv::Vec3d t = -(R_cw * C_world);
    cv::Affine3d P(R_cw, t);

    cv::Mat img = project2d(P, objects, 1280, 720, cv::Matx33d::eye());



    cv::imshow("render", img);
    cv::waitKey(0);

    return 0;
}
