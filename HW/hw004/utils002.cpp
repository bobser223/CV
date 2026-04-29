#include "utils002.h"
#include <random>
#include <iostream>

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
    const cv::Matx33d& cameraMatrix2) {

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

    double t_cam1_cam2_raw[] = {t_cam1_cam2_est[0], t_cam1_cam2_est[1], t_cam1_cam2_est[2]};


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



