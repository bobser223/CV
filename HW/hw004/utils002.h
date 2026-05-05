//
// Created by Volodymyr Avvakumov on 20.04.2026.
//

#ifndef CODE_UTILS002_H
#define CODE_UTILS002_H

#include <string>
#include <tuple>
#include <vector>

#include <ceres/ceres.h>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "cv_types.h"

std::vector<vec2> projectPoints(const std::vector<point3>& points,
                                  const cv::Affine3d affine,
                                  const cv::Matx33d& cameraMatrix);

std::vector<cv::Vec2d> normalizePixels(const std::vector<cv::Vec2d>& pixels, const cv::Matx33d& cameraMatrix);

cv::Affine3d getAffine(const std::vector<vec2>& x_pixels_pos_1, const std::vector<vec2>& x_pixels_pos_2,const cv::Matx33d& cameraMatrix);

std::vector<point3> triangulatePoints(
    const std::vector<vec2>& x_pixels_pos_1,
    const std::vector<vec2>& x_pixels_pos_2,
    const cv::Affine3d& affine,
    const cv::Matx33d& cameraMatrix);

std::tuple<cv::Matx33d, vec3, std::vector<point3>>estimateSLAM(const std::vector<vec2>& x_pixels_pos_1,
    const std::vector<vec2>& x_pixels_pos_2,
    const cv::Matx33d& cameraMatrix);

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

Eigen::Quaterniond rotationMatrixToQuaternion(const cv::Matx33d& R_cv);

std::vector<Eigen::Vector2d> cvVector2dToEigenVector2d(const std::vector<cv::Vec2d>& x);

struct ReprojectionCam1Pose1 {
    Eigen::Vector2d x;
    ReprojectionCam1Pose1(const Eigen::Vector2d& x): x(x){}

    template<typename T>
    bool operator()(const T* const X_raw, T* residual) const {
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
        const T* const X_raw, T* residual) const {
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
        const T* const X_raw, T* residual) const {
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
        const T* const X_raw, T* residual) const {

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

struct BinocularSLAMResult {
    Eigen::Quaterniond q_cam1_cam2;
    Eigen::Vector3d t_cam1_cam2;

    Eigen::Quaterniond q_pose1_pose2;
    Eigen::Vector3d t_pose1_pose2;

    std::vector<Eigen::Vector3d> points3D;

    ceres::Solver::Summary summary;
};

BinocularSLAMResult binocularSLAM(
    const std::vector<vec2>& x_pixels_pos_1_cam_1,
    const std::vector<vec2>& x_pixels_pos_1_cam_2,
    const std::vector<vec2>& x_pixels_pos_2_cam_1,
    const std::vector<vec2>& x_pixels_pos_2_cam_2,
    const cv::Matx33d& cameraMatrix1,
    const cv::Matx33d& cameraMatrix2,
    const vec3 baseline = vec3{0.,0.,0.});

cv::Matx33d rodriguesToMatx(const cv::Vec3d& rvec);

void addGaussianNoise(std::vector<vec2>& points, double sigma_px);

void printVec3(const std::string& name, const vec3& v);

void printEigenVec3(const std::string& name, const Eigen::Vector3d& v);

void printQuaternion(const std::string& name, const Eigen::Quaterniond& q);

void printRotationMatrixFromQuaternion(const std::string& name,
                                       const Eigen::Quaterniond& q);


void savePoints3DToFile(
    const std::vector<Eigen::Vector3d>& points3D,
    const std::string& filename
);

std::vector<Eigen::Vector3d> loadPoints3DFromFile(
    const std::string& filename
);

void drawAndSaveFourKeypointImages(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& img3,
    const cv::Mat& img4,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::KeyPoint>& kp3,
    const std::vector<cv::KeyPoint>& kp4,
    const std::string& outputDir,
    const std::string& prefix,
    bool showImages
);


void drawAndSaveFourMatchedKeypointImages(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& img3,
    const cv::Mat& img4,
    const std::vector<cv::Point2f>& matchedPts1,
    const std::vector<cv::Point2f>& matchedPts2,
    const std::vector<cv::Point2f>& matchedPts3,
    const std::vector<cv::Point2f>& matchedPts4,
    const std::string& outputDir,
    const std::string& prefix,
    bool showImages
);


void testBinocularSLAMWithNoise();

struct FourViewMatches {
    std::vector<vec2> x1; // pos1_cam1
    std::vector<vec2> x2; // pos1_cam2
    std::vector<vec2> x3; // pos2_cam1
    std::vector<vec2> x4; // pos2_cam2
};

std::vector<cv::DMatch> matchDescriptorsKNN(
    const cv::Mat& des1,
    const cv::Mat& des2,
    double ratio = 0.75);

FourViewMatches findFourViewMatches(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& img3,
    const cv::Mat& img4,
    bool debugDrawAllKeypoints = false,
    bool debugDrawMatchedKeypoints = false,
    bool debugShowImages = false,
    const std::string& debugOutputDir = "",
    const std::string& debugPrefix = "four_view"
);

void task004();
#endif //CODE_UTILS002_H
