#include "utils002.h"

#include <fstream>
#include <random>
#include <iostream>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "data_path.h"



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

        point3 p(0.0, 0.0, 0.0);

        if (std::abs(W) > 1e-12) {
            p = point3(X / W, Y / W, Z / W);
        }

        points3D.push_back(p);
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

Eigen::Quaterniond rotationMatrixToQuaternion(const cv::Matx33d& R_cv) {
    Eigen::Matrix3d R;

    R << R_cv(0, 0), R_cv(0, 1), R_cv(0, 2),
         R_cv(1, 0), R_cv(1, 1), R_cv(1, 2),
         R_cv(2, 0), R_cv(2, 1), R_cv(2, 2);

    Eigen::Quaterniond q(R);

    q.normalize();

    return q;
}

std::vector<Eigen::Vector2d> cvVector2dToEigenVector2d(const std::vector<cv::Vec2d>& x) {
    std::vector<Eigen::Vector2d> res;
    res.reserve(x.size());
    for (const auto& vec2d: x) {
        res.emplace_back(vec2d[0], vec2d[1]);
    }
    return res;
}



BinocularSLAMResult binocularSLAM(
    const std::vector<vec2>& x_pixels_pos_1_cam_1,
    const std::vector<vec2>& x_pixels_pos_1_cam_2,
    const std::vector<vec2>& x_pixels_pos_2_cam_1,
    const std::vector<vec2>& x_pixels_pos_2_cam_2,
    const cv::Matx33d& cameraMatrix1,
    const cv::Matx33d& cameraMatrix2,
    const vec3 baseline) {

    auto x_norm_pos_1_cam_1 = cvVector2dToEigenVector2d( normalizePixels(x_pixels_pos_1_cam_1, cameraMatrix1));
    auto x_norm_pos_1_cam_2 = cvVector2dToEigenVector2d( normalizePixels(x_pixels_pos_1_cam_2, cameraMatrix2));
    auto x_norm_pos_2_cam_1 = cvVector2dToEigenVector2d( normalizePixels(x_pixels_pos_2_cam_1, cameraMatrix1));
    auto x_norm_pos_2_cam_2 =  cvVector2dToEigenVector2d(normalizePixels(x_pixels_pos_2_cam_2, cameraMatrix2));

    auto [R_cam1_cam2_est, t_cam1_cam2_est, points3D_stereo_est] =
    estimateSLAM(
        x_pixels_pos_1_cam_1,
        x_pixels_pos_1_cam_2,
        cameraMatrix1
    );
    Eigen::Quaterniond q_cam1_cam2 = rotationMatrixToQuaternion(R_cam1_cam2_est);
    double q_cam1_cam2_raw[] = {
        q_cam1_cam2.x(),
        q_cam1_cam2.y(),
        q_cam1_cam2.z(),
        q_cam1_cam2.w()
    };

    double t_cam1_cam2_raw[3];

    if (baseline.dot(baseline) < 1e-9) { // vector == 0, default parameter
        t_cam1_cam2_raw[0] = t_cam1_cam2_est[0];
        t_cam1_cam2_raw[1] = t_cam1_cam2_est[1];
        t_cam1_cam2_raw[2] = t_cam1_cam2_est[2];
    } else {
        t_cam1_cam2_raw[0] = baseline[0];
        t_cam1_cam2_raw[1] = baseline[1];
        t_cam1_cam2_raw[2] = baseline[2];
    }



    auto [R_pose1_pose2_est, t_pose1_pose2_est, points3D_motion_est] =
        estimateSLAM(
            x_pixels_pos_1_cam_1,
            x_pixels_pos_2_cam_1,
            cameraMatrix1
        );

    Eigen::Quaterniond q_pose1_pose2 = rotationMatrixToQuaternion(R_pose1_pose2_est);
    double q_pose1_pose2_raw[] = {
        q_pose1_pose2.x(),
        q_pose1_pose2.y(),
        q_pose1_pose2.z(),
        q_pose1_pose2.w()
    };

    double t_pose1_pose2_raw[] = {t_pose1_pose2_est[0], t_pose1_pose2_est[1], t_pose1_pose2_est[2]};

    // cv::Matx33d R_cam2_pose2 =
    //     R_cam1_cam2_est * R_pose1_pose2_est;
    //
    // vec3 t_cam2_pose2 =
    //     R_cam1_cam2_est * t_pose1_pose2_est + t_cam1_cam2_est;


    std::vector<Eigen::Vector3d> X_est;
    X_est.reserve(points3D_stereo_est.size());

    for (const auto& p : points3D_stereo_est) {
        X_est.emplace_back(p.x, p.y, p.z);
    }

    ceres::Problem problem;

    CV_Assert(x_norm_pos_1_cam_1.size() == X_est.size());
    CV_Assert(x_norm_pos_1_cam_2.size() == X_est.size());
    CV_Assert(x_norm_pos_2_cam_1.size() == X_est.size());
    CV_Assert(x_norm_pos_2_cam_2.size() == X_est.size());

    for (size_t i = 0; i < X_est.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionCam1Pose1, 2, 3>(
                new ReprojectionCam1Pose1(x_norm_pos_1_cam_1[i])
            ),
            nullptr,
            X_est[i].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionCam2Pose1, 2, 4, 3, 3>(
                new ReprojectionCam2Pose1(x_norm_pos_1_cam_2[i])
            ),
            nullptr,
            q_cam1_cam2_raw,
            t_cam1_cam2_raw,
            X_est[i].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionCam1Pose2, 2, 4, 3, 3>(
                new ReprojectionCam1Pose2(x_norm_pos_2_cam_1[i])
            ),
            nullptr,
            q_pose1_pose2_raw,
            t_pose1_pose2_raw,
            X_est[i].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionCam2Pose2, 2, 4, 3, 4, 3, 3>(
                new ReprojectionCam2Pose2(x_norm_pos_2_cam_2[i])
            ),
            nullptr,
            q_cam1_cam2_raw,
            t_cam1_cam2_raw,
            q_pose1_pose2_raw,
            t_pose1_pose2_raw,
            X_est[i].data()
        );
    }

    problem.SetManifold(
        q_cam1_cam2_raw,
        new ceres::EigenQuaternionManifold
    );

    problem.SetManifold(
        q_pose1_pose2_raw,
        new ceres::EigenQuaternionManifold
    );

    problem.SetParameterBlockConstant(t_cam1_cam2_raw);
    // problem.SetParameterBlockConstant(q_cam1_cam2_raw);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    Eigen::Quaterniond q_cam1_cam2_final(
    q_cam1_cam2_raw[3],
    q_cam1_cam2_raw[0],
    q_cam1_cam2_raw[1],
    q_cam1_cam2_raw[2]
);

    Eigen::Vector3d t_cam1_cam2_final(
        t_cam1_cam2_raw[0],
        t_cam1_cam2_raw[1],
        t_cam1_cam2_raw[2]
    );

    Eigen::Quaterniond q_pose1_pose2_final(
        q_pose1_pose2_raw[3],
        q_pose1_pose2_raw[0],
        q_pose1_pose2_raw[1],
        q_pose1_pose2_raw[2]
    );

    Eigen::Vector3d t_pose1_pose2_final(
        t_pose1_pose2_raw[0],
        t_pose1_pose2_raw[1],
        t_pose1_pose2_raw[2]
    );

    return BinocularSLAMResult{
        q_cam1_cam2_final,
        t_cam1_cam2_final,
        q_pose1_pose2_final,
        t_pose1_pose2_final,
        X_est,
        summary
    };





}


cv::Matx33d rodriguesToMatx(const cv::Vec3d& rvec) {
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);

    return cv::Matx33d(
        R_mat.at<double>(0, 0), R_mat.at<double>(0, 1), R_mat.at<double>(0, 2),
        R_mat.at<double>(1, 0), R_mat.at<double>(1, 1), R_mat.at<double>(1, 2),
        R_mat.at<double>(2, 0), R_mat.at<double>(2, 1), R_mat.at<double>(2, 2)
    );
}

void addGaussianNoise(std::vector<vec2>& points, double sigma_px) {
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, sigma_px);

    for (auto& p : points) {
        p[0] += noise(gen);
        p[1] += noise(gen);
    }
}

void printVec3(const std::string& name, const vec3& v) {
    std::cout << name << " = ["
              << v[0] << ", "
              << v[1] << ", "
              << v[2] << "]\n";
}

void printQuaternion(const std::string& name, const Eigen::Quaterniond& q) {
    std::cout << name << " = [x y z w] = ["
              << q.x() << ", "
              << q.y() << ", "
              << q.z() << ", "
              << q.w() << "]\n";
}

void printEigenVec3(const std::string& name, const Eigen::Vector3d& v) {
    std::cout << name << " = ["
              << v.x() << ", "
              << v.y() << ", "
              << v.z() << "]\n";
}

void printRotationMatrixFromQuaternion(const std::string& name,
                                       const Eigen::Quaterniond& q) {
    std::cout << name << ":\n"
              << q.normalized().toRotationMatrix()
              << "\n";
}

void testBinocularSLAMWithNoise() {

    cv::Matx33d K(
        800.0,   0.0, 320.0,
          0.0, 800.0, 240.0,
          0.0,   0.0,   1.0
    );

    cv::Matx33d cameraMatrix1 = K;
    cv::Matx33d cameraMatrix2 = K;


    std::vector<point3> points_true = {
        {-1.2, -0.8, 5.0},
        {-0.4, -0.7, 4.5},
        { 0.5, -0.6, 5.5},
        { 1.1, -0.4, 6.0},
        {-1.0,  0.2, 6.5},
        {-0.2,  0.1, 5.2},
        { 0.7,  0.2, 4.8},
        { 1.4,  0.4, 6.3},
        {-0.8,  0.9, 7.0},
        { 0.1,  0.8, 6.8},
        { 0.9,  0.9, 7.2},
        {-1.5, -0.1, 8.0},
        { 1.6, -0.2, 7.5},
        {-0.6,  1.2, 8.5},
        { 0.6, -1.0, 7.8}
    };

    cv::Matx33d R_cam1_pose1 = cv::Matx33d::eye();
    vec3 t_cam1_pose1(0.0, 0.0, 0.0);

    cv::Affine3d affine_cam1_pose1(
        R_cam1_pose1,
        t_cam1_pose1
    );

    cv::Vec3d rvec_cam1_cam2_true(0.005, -0.015, 0.003);
    cv::Matx33d R_cam1_cam2_true = rodriguesToMatx(rvec_cam1_cam2_true);
    vec3 t_cam1_cam2_true(-1.0, 0.0, 0.0);

    cv::Affine3d affine_cam2_pose1(
        R_cam1_cam2_true,
        t_cam1_cam2_true
    );


    cv::Vec3d rvec_pose1_pose2_true(0.03, -0.06, 0.02);
    cv::Matx33d R_pose1_pose2_true = rodriguesToMatx(rvec_pose1_pose2_true);
    vec3 t_pose1_pose2_true(-0.35, 0.05, 0.10);

    cv::Affine3d affine_cam1_pose2(
        R_pose1_pose2_true,
        t_pose1_pose2_true
    );

    // 5. Cam2 Pose2 = stereo * motion.
    //
    // X_cam1_pose2 = R_pose * X + t_pose
    // X_cam2_pose2 = R_stereo * X_cam1_pose2 + t_stereo
    //
    // R = R_stereo * R_pose
    // t = R_stereo * t_pose + t_stereo

    cv::Matx33d R_cam2_pose2_true =
        R_cam1_cam2_true * R_pose1_pose2_true;

    vec3 t_cam2_pose2_true =
        R_cam1_cam2_true * t_pose1_pose2_true + t_cam1_cam2_true;

    cv::Affine3d affine_cam2_pose2(
        R_cam2_pose2_true,
        t_cam2_pose2_true
    );

    std::vector<vec2> x_pixels_pos_1_cam_1 =
        projectPoints(points_true, affine_cam1_pose1, cameraMatrix1);

    std::vector<vec2> x_pixels_pos_1_cam_2 =
        projectPoints(points_true, affine_cam2_pose1, cameraMatrix2);

    std::vector<vec2> x_pixels_pos_2_cam_1 =
        projectPoints(points_true, affine_cam1_pose2, cameraMatrix1);

    std::vector<vec2> x_pixels_pos_2_cam_2 =
        projectPoints(points_true, affine_cam2_pose2, cameraMatrix2);

    CV_Assert(x_pixels_pos_1_cam_1.size() == points_true.size());
    CV_Assert(x_pixels_pos_1_cam_2.size() == points_true.size());
    CV_Assert(x_pixels_pos_2_cam_1.size() == points_true.size());
    CV_Assert(x_pixels_pos_2_cam_2.size() == points_true.size());

   // noise
    double sigma_px = 0.5;

    addGaussianNoise(x_pixels_pos_1_cam_1, sigma_px);
    addGaussianNoise(x_pixels_pos_1_cam_2, sigma_px);
    addGaussianNoise(x_pixels_pos_2_cam_1, sigma_px);
    addGaussianNoise(x_pixels_pos_2_cam_2, sigma_px);

    std::cout << "\n===== TRUE PARAMETERS =====\n";
    std::cout << "R_cam1_cam2_true:\n" << cv::Mat(R_cam1_cam2_true) << "\n";
    printVec3("t_cam1_cam2_true", t_cam1_cam2_true);

    std::cout << "\nR_pose1_pose2_true:\n" << cv::Mat(R_pose1_pose2_true) << "\n";
    printVec3("t_pose1_pose2_true", t_pose1_pose2_true);

    std::cout << "\nR_cam2_pose2_true:\n" << cv::Mat(R_cam2_pose2_true) << "\n";
    printVec3("t_cam2_pose2_true", t_cam2_pose2_true);

    std::cout << "\n===== FIRST FEW PIXEL OBSERVATIONS =====\n";
    for (size_t i = 0; i < std::min<size_t>(5, points_true.size()); ++i) {
        std::cout << "point " << i << "\n";
        std::cout << "  cam1 pose1 = " << x_pixels_pos_1_cam_1[i] << "\n";
        std::cout << "  cam2 pose1 = " << x_pixels_pos_1_cam_2[i] << "\n";
        std::cout << "  cam1 pose2 = " << x_pixels_pos_2_cam_1[i] << "\n";
        std::cout << "  cam2 pose2 = " << x_pixels_pos_2_cam_2[i] << "\n";
    }

    BinocularSLAMResult result = binocularSLAM(
    x_pixels_pos_1_cam_1,
    x_pixels_pos_1_cam_2,
    x_pixels_pos_2_cam_1,
    x_pixels_pos_2_cam_2,
    cameraMatrix1,
    cameraMatrix2
);

    std::cout << "\n===== CERES SUMMARY =====\n";
    std::cout << result.summary.BriefReport() << "\n";

    std::cout << "\n===== ESTIMATED PARAMETERS =====\n";

    printQuaternion("q_cam1_cam2_est", result.q_cam1_cam2);
    printEigenVec3("t_cam1_cam2_est", result.t_cam1_cam2);
    printRotationMatrixFromQuaternion("R_cam1_cam2_est", result.q_cam1_cam2);

    std::cout << "\n";

    printQuaternion("q_pose1_pose2_est", result.q_pose1_pose2);
    printEigenVec3("t_pose1_pose2_est", result.t_pose1_pose2);
    printRotationMatrixFromQuaternion("R_pose1_pose2_est", result.q_pose1_pose2);

    std::cout << "\n===== ESTIMATED POINTS =====\n";
    for (size_t i = 0; i < result.points3D.size(); ++i) {
        std::cout << "X_est[" << i << "] = "
                  << result.points3D[i].transpose()
                  << "\n";
    }
}

std::vector<cv::DMatch> matchDescriptorsKNN(
    const cv::Mat& des1,
    const cv::Mat& des2,
    double ratio
) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(des1, des2, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;

    for (const auto& pair : knnMatches) {
        if (pair.size() < 2) {
            continue;
        }

        const cv::DMatch& m = pair[0];
        const cv::DMatch& n = pair[1];

        if (m.distance < ratio * n.distance) {
            goodMatches.push_back(m);
        }
    }

    return goodMatches;
}


void savePoints3DToFile(
    const std::vector<Eigen::Vector3d>& points3D,
    const std::string& filename
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << std::setprecision(17);

    for (const auto& p : points3D) {
        file << p.x() << " "
             << p.y() << " "
             << p.z() << "\n";
    }

    file.close();
}


std::vector<Eigen::Vector3d> loadPoints3DFromFile(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::vector<Eigen::Vector3d> points3D;

    double x, y, z;

    while (file >> x >> y >> z) {
        points3D.emplace_back(x, y, z);
    }

    file.close();

    return points3D;
}

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
) {
    cv::Mat out1, out2, out3, out4;

    cv::drawKeypoints(
        img1, kp1, out1,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cv::drawKeypoints(
        img2, kp2, out2,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cv::drawKeypoints(
        img3, kp3, out3,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cv::drawKeypoints(
        img4, kp4, out4,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    if (!outputDir.empty()) {
        cv::imwrite(outputDir + prefix + "_img1_all_keypoints.jpg", out1);
        cv::imwrite(outputDir + prefix + "_img2_all_keypoints.jpg", out2);
        cv::imwrite(outputDir + prefix + "_img3_all_keypoints.jpg", out3);
        cv::imwrite(outputDir + prefix + "_img4_all_keypoints.jpg", out4);
    }

    if (showImages) {
        cv::imshow(prefix + " img1 all keypoints", out1);
        cv::imshow(prefix + " img2 all keypoints", out2);
        cv::imshow(prefix + " img3 all keypoints", out3);
        cv::imshow(prefix + " img4 all keypoints", out4);
        cv::waitKey(0);
    }
}

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
) {
    cv::Mat out1 = img1.clone();
    cv::Mat out2 = img2.clone();
    cv::Mat out3 = img3.clone();
    cv::Mat out4 = img4.clone();

    for (size_t i = 0; i < matchedPts1.size(); ++i) {
        cv::circle(out1, matchedPts1[i], 8, cv::Scalar(0, 0, 255), 2);
        cv::circle(out2, matchedPts2[i], 8, cv::Scalar(0, 0, 255), 2);
        cv::circle(out3, matchedPts3[i], 8, cv::Scalar(0, 0, 255), 2);
        cv::circle(out4, matchedPts4[i], 8, cv::Scalar(0, 0, 255), 2);

        cv::putText(
            out1,
            std::to_string(i),
            matchedPts1[i] + cv::Point2f(5.0f, -5.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1
        );

        cv::putText(
            out2,
            std::to_string(i),
            matchedPts2[i] + cv::Point2f(5.0f, -5.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1
        );

        cv::putText(
            out3,
            std::to_string(i),
            matchedPts3[i] + cv::Point2f(5.0f, -5.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1
        );

        cv::putText(
            out4,
            std::to_string(i),
            matchedPts4[i] + cv::Point2f(5.0f, -5.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1
        );
    }

    if (!outputDir.empty()) {
        cv::imwrite(outputDir + prefix + "_img1_common_matches.jpg", out1);
        cv::imwrite(outputDir + prefix + "_img2_common_matches.jpg", out2);
        cv::imwrite(outputDir + prefix + "_img3_common_matches.jpg", out3);
        cv::imwrite(outputDir + prefix + "_img4_common_matches.jpg", out4);
    }

    if (showImages) {
        cv::imshow(prefix + " img1 common matches", out1);
        cv::imshow(prefix + " img2 common matches", out2);
        cv::imshow(prefix + " img3 common matches", out3);
        cv::imshow(prefix + " img4 common matches", out4);
        cv::waitKey(0);
    }
}





FourViewMatches findFourViewMatches(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& img3,
    const cv::Mat& img4,
    bool debugDrawAllKeypoints ,
    bool debugDrawMatchedKeypoints ,
    bool debugShowImages ,
    const std::string& debugOutputDir ,
    const std::string& debugPrefix
) {
    cv::Mat gray1, gray2, gray3, gray4;

    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img3, gray3, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img4, gray4, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        50000,
        1.2f,
        8,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        20
    );

    std::vector<cv::KeyPoint> kp1, kp2, kp3, kp4;
    cv::Mat des1, des2, des3, des4;

    orb->detectAndCompute(gray1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(gray2, cv::noArray(), kp2, des2);
    orb->detectAndCompute(gray3, cv::noArray(), kp3, des3);
    orb->detectAndCompute(gray4, cv::noArray(), kp4, des4);

    if (des1.empty() || des2.empty() || des3.empty() || des4.empty()) {
        throw std::runtime_error("Some image has no ORB descriptors");
    }

    std::cout << "Keypoints:\n";
    std::cout << "img1: " << kp1.size() << "\n";
    std::cout << "img2: " << kp2.size() << "\n";
    std::cout << "img3: " << kp3.size() << "\n";
    std::cout << "img4: " << kp4.size() << "\n";

    if (debugDrawAllKeypoints) {
        drawAndSaveFourKeypointImages(
            img1, img2, img3, img4,
            kp1, kp2, kp3, kp4,
            debugOutputDir,
            debugPrefix,
            debugShowImages
        );
    }

    std::vector<cv::DMatch> matches12 = matchDescriptorsKNN(des1, des2, 0.75);
    std::vector<cv::DMatch> matches13 = matchDescriptorsKNN(des1, des3, 0.75);
    std::vector<cv::DMatch> matches14 = matchDescriptorsKNN(des1, des4, 0.75);

    std::cout << "Pair matches:\n";
    std::cout << "1-2: " << matches12.size() << "\n";
    std::cout << "1-3: " << matches13.size() << "\n";
    std::cout << "1-4: " << matches14.size() << "\n";

    std::unordered_map<int, int> map12;
    std::unordered_map<int, int> map13;
    std::unordered_map<int, int> map14;

    for (const auto& m : matches12) {
        map12[m.queryIdx] = m.trainIdx;
    }

    for (const auto& m : matches13) {
        map13[m.queryIdx] = m.trainIdx;
    }

    for (const auto& m : matches14) {
        map14[m.queryIdx] = m.trainIdx;
    }

    FourViewMatches result;

    std::vector<cv::Point2f> matchedPts1;
    std::vector<cv::Point2f> matchedPts2;
    std::vector<cv::Point2f> matchedPts3;
    std::vector<cv::Point2f> matchedPts4;

    for (const auto& [idx1, idx2] : map12) {
        auto it13 = map13.find(idx1);
        auto it14 = map14.find(idx1);

        if (it13 == map13.end() || it14 == map14.end()) {
            continue;
        }

        int idx3 = it13->second;
        int idx4 = it14->second;

        cv::Point2f p1 = kp1[idx1].pt;
        cv::Point2f p2 = kp2[idx2].pt;
        cv::Point2f p3 = kp3[idx3].pt;
        cv::Point2f p4 = kp4[idx4].pt;

        result.x1.emplace_back(p1.x, p1.y);
        result.x2.emplace_back(p2.x, p2.y);
        result.x3.emplace_back(p3.x, p3.y);
        result.x4.emplace_back(p4.x, p4.y);

        matchedPts1.push_back(p1);
        matchedPts2.push_back(p2);
        matchedPts3.push_back(p3);
        matchedPts4.push_back(p4);
    }

    std::cout << "Four-view common matches: " << result.x1.size() << "\n";

    if (debugDrawMatchedKeypoints) {
        drawAndSaveFourMatchedKeypointImages(
            img1, img2, img3, img4,
            matchedPts1, matchedPts2, matchedPts3, matchedPts4,
            debugOutputDir,
            debugPrefix,
            debugShowImages
        );
    }

    return result;
}

void task004() {
        cv::Matx33d K(
        4084.54110, 0.0,        2836.77733,
        0.0,        4096.24707, 2139.91516,
        0.0,        0.0,        1.0
    );

    cv::Vec<double, 5> distCoeffs(
         0.20924062,
        -0.64274374,
        -0.00404513,
        -0.00353257,
         0.36497480
    );

    std::string path_to_images = PATH_TO_DATA + "hw004/sheep_iphone_17_pro_main/jpg/";
    cv::Mat pos1_cam1_img = cv::imread(path_to_images + "pos1_cam1.jpg");
    cv::Mat pos1_cam2_img = cv::imread(path_to_images + "pos1_cam2.jpg");
    cv::Mat pos2_cam1_img = cv::imread(path_to_images + "pos2_cam1.jpg");
    cv::Mat pos2_cam2_img = cv::imread(path_to_images + "pos2_cam2.jpg");

    auto [pos1_cam1, pos1_cam2, pos2_cam1, pos2_cam2] =
        findFourViewMatches(
            pos1_cam1_img,
            pos1_cam2_img,
            pos2_cam1_img,
            pos2_cam2_img,
            true,
            true,
            true,
            PATH_TO_DATA + "hw004/sheep_iphone_17_pro_main/debug/"
        );

    auto orb = cv::ORB::create(2000, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);

    auto [ q_cam1_cam2,
     t_cam1_cam2,
    q_pose1_pose2,
    t_pose1_pose2,
    points3D,
    summary] = binocularSLAM(pos1_cam1, pos1_cam2, pos2_cam1, pos2_cam2, K, K, {-0.3,0.,0.});

    std::cout << "q_cam1_cam2 = "
              << "[w: " << q_cam1_cam2.w()
              << ", x: " << q_cam1_cam2.x()
              << ", y: " << q_cam1_cam2.y()
              << ", z: " << q_cam1_cam2.z()
              << "]\n";

    std::cout << "t_cam1_cam2 = "
              << t_cam1_cam2.transpose()
              << "\n\n";

    std::cout << "q_pose1_pose2 = "
              << "[w: " << q_pose1_pose2.w()
              << ", x: " << q_pose1_pose2.x()
              << ", y: " << q_pose1_pose2.y()
              << ", z: " << q_pose1_pose2.z()
              << "]\n";

    std::cout << "t_pose1_pose2 = "
              << t_pose1_pose2.transpose()
              << "\n";

    savePoints3DToFile(points3D, PATH_TO_DATA + "hw004/sheep_iphone_17_pro_main/points3D.txt");




}

