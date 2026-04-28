#include "utils002.h"


#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>



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

struct BundleAdjustment {
    Eigen::Vector2d x;

    BundleAdjustment(const Eigen::Vector2d& x): x(x)
    {}

    template <typename T>
    bool operator()(const T* const q_0,const T* const t_0, const T* const X_0, T* residuals) const {

        Eigen::Quaternion<T> q(q_0[3], q_0[0], q_0[1], q_0[2]);

        Eigen::Matrix<T, 3, 1> t(
            t_0[0],
            t_0[1],
            t_0[2]
        );

        Eigen::Matrix<T, 3, 1> X(
            X_0[0],
            X_0[1],
            X_0[2]
        );
        Eigen::Matrix<T, 3, 1> X_cam = q * X + t;

        T u = X_cam[0] / X_cam[2];
        T v = X_cam[1] / X_cam[2];

        residuals[0] = u - T(x[0]);
        residuals[1] = v - T(x[1]);

        return true;
    }


};


struct ReprojectionCam1Pose1 {
    Eigen::Vector2d x;
    ReprojectionCam1Pose1(const Eigen::Vector2d& x): x(x){}

    template<typename T>
    bool operator()(const T* const X_raw, T* residual) {
        Eigen::Matrix<T, 3, 1> X(X_raw[0], X_raw[1], X_raw[2]);
        T u = X[0] / X[2];
        T v = X[1] / X[2];
        residual[0] = u - T(x[0]);
        residual[1] = v - T(x[1]);
        return true;
    }
};

struct ReprojectionCam2Pose1{
    Eigen::Vector2d x;
    ReprojectionCam2Pose1(const Eigen::Vector2d& x): x(x){}

    template<typename T>
    bool operator()(const T* const q_cam1_cam2_raw,const T* const t_cam1_cam2_raw,
        const T* const X_raw, T* residual) {
        Eigen::Quaternion<T> q_cam1_cam2(q_cam1_cam2_raw[3],
            q_cam1_cam2_raw[0],
            q_cam1_cam2_raw[1],
            q_cam1_cam2_raw[2]);

        Eigen::Matrix<T, 3, 1> t_cam1_cam2(
            t_cam1_cam2_raw[0],
            t_cam1_cam2_raw[1],
            t_cam1_cam2_raw[2]
        );

        Eigen::Matrix<T, 3, 1> X(
            X_raw[0],
            X_raw[1],
            X_raw[2]
        );

        Eigen::Matrix<T, 3, 1> X_cam1_cam2 = q_cam1_cam2 * X + t_cam1_cam2;

        T u = X_cam1_cam2[0] / X_cam1_cam2[2];
        T v = X_cam1_cam2[1] / X_cam1_cam2[2];
        residual[0] = u - T(x[0]);
        residual[1] = v - T(x[1]);
        return true;
    }
};

struct ReprojectionCam1Pose2{
    Eigen::Vector2d x;
    ReprojectionCam1Pose2(const Eigen::Vector2d& x): x(x){}

    template<typename T>
    bool operator()(const T* const q_pose1_pose2_raw ,const T* const t_pose1_pose2_raw,
        const T* const X_raw, T* residual) {
        Eigen::Quaternion<T> q_pose1_pose2(q_pose1_pose2_raw[3], q_pose1_pose2_raw[0], q_pose1_pose2_raw[1], q_pose1_pose2_raw[2]);

        Eigen::Matrix<T, 3, 1> t_pose1_pose2(
            t_pose1_pose2_raw[0],
            t_pose1_pose2_raw[1],
            t_pose1_pose2_raw[2]
            );


        Eigen::Matrix<T, 3, 1> X(
            X_raw[0],
            X_raw[1],
            X_raw[2]
        );

        Eigen::Matrix<T, 3, 1> X_cam1_pose2 = q_pose1_pose2 * X + t_pose1_pose2;

        T u = X_cam1_pose2[0] / X_cam1_pose2[2];
        T v = X_cam1_pose2[1] / X_cam1_pose2[2];
        residual[0] = u - T(x[0]);
        residual[1] = v - T(x[1]);
        return true;
    }
};

struct ReprojectionCam2Pose2{
    Eigen::Vector2d x;
    ReprojectionCam2Pose2(const Eigen::Vector2d& x): x(x){}


    template<typename T>
    bool operator()(const T* const q_cam1_cam2_raw,const T* const t_cam1_cam2_raw,
        const T* const q_pose1_pose2_raw ,const T* const t_pose1_pose2_raw,
        const T* const X_raw, T* residual) {

        Eigen::Quaternion<T> q_cam1_cam2(q_cam1_cam2_raw[3],
            q_cam1_cam2_raw[0],
            q_cam1_cam2_raw[1],
            q_cam1_cam2_raw[2]);

        Eigen::Matrix<T, 3, 1> t_cam1_cam2(
            t_cam1_cam2_raw[0],
            t_cam1_cam2_raw[1],
            t_cam1_cam2_raw[2]
        );

        Eigen::Quaternion<T> q_pose1_pose2(q_pose1_pose2_raw[3], q_pose1_pose2_raw[0], q_pose1_pose2_raw[1], q_pose1_pose2_raw[2]);

        Eigen::Matrix<T, 3, 1> t_pose1_pose2(
            t_pose1_pose2_raw[0],
            t_pose1_pose2_raw[1],
            t_pose1_pose2_raw[2]
            );


        Eigen::Matrix<T, 3, 1> X(
            X_raw[0],
            X_raw[1],
            X_raw[2]
        );

        auto X_cam2_pose2 =q_cam1_cam2 * (q_pose1_pose2 * X + t_pose1_pose2) + t_cam1_cam2;

        T u = X_cam2_pose2[0] / X_cam2_pose2[2];
        T v = X_cam2_pose2[1] / X_cam2_pose2[2];
        residual[0] = u - T(x[0]);
        residual[1] = v - T(x[1]);
        return true;
    }



};




auto binocularSLAM(
    const std::vector<vec2>& x_pixels_pos_1_cam_1,
    const std::vector<vec2>& x_pixels_pos_1_cam_2,
    const std::vector<vec2>& x_pixels_pos_2_cam_1,
    const std::vector<vec2>& x_pixels_pos_2_cam_2,
    const cv::Matx33d& cameraMatrix1,
    const cv::Matx33d& cameraMatrix2) {

    // estimation block
    auto [R_1, t_1, points3D] =
        estimateSLAM(x_pixels_pos_1_cam_1, x_pixels_pos_1_cam_2, cameraMatrix1);

    auto [R_2, t_2, points3D_2] =
        estimateSLAM(x_pixels_pos_1_cam_1, x_pixels_pos_2_cam_1, cameraMatrix2); // R2(r1+t1) + t2

    auto [R3, t3, points3D_3] =
        estimateSLAM(x_pixels_pos_1_cam_1, x_pixels_pos_2_cam_2, cameraMatrix2);




}





