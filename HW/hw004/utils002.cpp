#include "utils002.h"

#include <opencv2/calib3d.hpp>



std::vector<vec2> projectPoints(const std::vector<point3>& points,
                                  const cv::Affine3d affine,
                                  const cv::Matx33d& cameraMatrix) {
    cv::Matx33d R = affine.rotation();
    vec3 t = affine.translation();

    std::vector<vec2> projected;
    projected.reserve(points.size());

    for (const auto& pointNED : points) {
        vec3 p(pointNED.x, pointNED.y, pointNED.z);
        vec3 point_local = R * p + t;

        if (point_local[2] <= 1e-9) {
            continue;
        }

        double u = point_local[0] / point_local[2];
        double v = point_local[1] / point_local[2];

        double px = cameraMatrix(0, 0) * u + cameraMatrix(0, 2);
        double py = cameraMatrix(1, 1) * v + cameraMatrix(1, 2);

        projected.emplace_back(px, py);
    }

    return projected;
}


std::vector<cv::Vec2d> normalizePixels(const std::vector<cv::Vec2d>& pixels, const cv::Matx33d& cameraMatrix) {
    cv::Matx33d K_inv = cameraMatrix.inv();
    std::vector<cv::Vec2d> normalized;

    for (const auto& p : pixels) {
        cv::Vec3d ph(p[0], p[1], 1.0);
        cv::Vec3d pn = K_inv * ph;
        normalized.emplace_back(pn[0] / pn[2], pn[1] / pn[2]);
    }

    return normalized;
}

cv::Affine3d getAffine(const std::vector<vec2>& x_pixels_pos_1, const std::vector<vec2>& x_pixels_pos_2,const cv::Matx33d& cameraMatrix) {
    cv::Mat E_cv = cv::findEssentialMat(x_pixels_pos_1, x_pixels_pos_2, cameraMatrix);
    cv::Mat R_from_cv_function, t_from_cv_function;
    cv::recoverPose(E_cv, x_pixels_pos_1, x_pixels_pos_2, cameraMatrix, R_from_cv_function, t_from_cv_function);
    return cv::Affine3d(R_from_cv_function, t_from_cv_function);
}

std::vector<point3> triangulatePoints(
    const std::vector<vec2>& x_pixels_pos_1,
    const std::vector<vec2>& x_pixels_pos_2,
    const cv::Affine3d& affine,
    const cv::Matx33d& cameraMatrix
) {
    CV_Assert(x_pixels_pos_1.size() == x_pixels_pos_2.size());

    std::vector<point3> points3D;
    points3D.reserve(x_pixels_pos_1.size());

    // pixel -> K^{-1} * pixel
    std::vector<vec2> x_norm_1 = normalizePixels(x_pixels_pos_1, cameraMatrix);
    std::vector<vec2> x_norm_2 = normalizePixels(x_pixels_pos_2, cameraMatrix);



    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    P1.at<double>(0, 0) = 1.0;
    P1.at<double>(1, 1) = 1.0;
    P1.at<double>(2, 2) = 1.0;

    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);

    cv::Matx33d R = affine.rotation();
    vec3 t = affine.translation();

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            P2.at<double>(r, c) = R(r, c);
        }
        P2.at<double>(r, 3) = t[r];
    }

    cv::Mat pts1(2, static_cast<int>(x_norm_1.size()), CV_64F);
    cv::Mat pts2(2, static_cast<int>(x_norm_2.size()), CV_64F);

    for (int i = 0; i < static_cast<int>(x_norm_1.size()); ++i) {
        pts1.at<double>(0, i) = x_norm_1[i][0];
        pts1.at<double>(1, i) = x_norm_1[i][1];

        pts2.at<double>(0, i) = x_norm_2[i][0];
        pts2.at<double>(1, i) = x_norm_2[i][1];
    }


    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);



    for (int i = 0; i < points4D.cols; ++i) {
        double X = points4D.at<double>(0, i);
        double Y = points4D.at<double>(1, i);
        double Z = points4D.at<double>(2, i);
        double W = points4D.at<double>(3, i);

        if (std::abs(W) < 1e-12) {
            continue;
        }

        point3 p(
            X / W,
            Y / W,
            Z / W
        );


        if (p.z > 0.0) {
            points3D.push_back(p);
        }
    }

    return points3D;
}

std::tuple<cv::Matx33d, vec3, std::vector<point3>>estimateSLAM(const std::vector<vec2>& x_pixels_pos_1,
    const std::vector<vec2>& x_pixels_pos_2,
    const cv::Matx33d& cameraMatrix) {
    cv::Affine3d affine = getAffine(x_pixels_pos_1, x_pixels_pos_2, cameraMatrix);
    std::vector<point3> points3D = triangulatePoints(x_pixels_pos_1, x_pixels_pos_2, affine, cameraMatrix);
    return {affine.rotation(), affine.translation(), points3D};
}






