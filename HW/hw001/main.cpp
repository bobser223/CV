#include <iostream>

#include "scane.h"


int main() {
    auto objects = createObjects();


    const int W=1280, H=720;
    cv::Matx33d K(
        800, 0, W/2.0,
        0, 800, H/2.0,
        0,   0,   1
    );

    cv::Vec3d T_world(10306, 306, -304);        // target (center of apple)
    cv::Vec3d C_world(10306 - 2000, 306, -304 - 500); // camera a bit back in North and up (more negative Z)
    cv::Vec3d upHint_world(0, 0, -1);           // up in NED

    auto lookAt_R_cw = [](const cv::Vec3d& C, const cv::Vec3d& T, const cv::Vec3d& upHint) {
        cv::Vec3d f = T - C; f *= 1.0 / cv::norm(f);   // forward (world)
        cv::Vec3d r = f.cross(upHint); r *= 1.0 / cv::norm(r);
        cv::Vec3d u = r.cross(f);
        return cv::Matx33d(
            r[0], r[1], r[2],
            u[0], u[1], u[2],
            f[0], f[1], f[2]
        );
    };

    cv::Matx33d R = lookAt_R_cw(C_world, T_world, upHint_world);
    cv::Vec3d t = -(R * C_world);
    cv::Affine3d P(R, t);

    cv::Mat img = project2d(P, objects, W, H, K);



    cv::imshow("render", img);
    cv::waitKey(0);

    return 0;
}
