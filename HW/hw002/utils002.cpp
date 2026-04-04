//
// Created by Volodymyr Avvakumov on 17.03.2026.
//

#include "utils002.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/affine.hpp>

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

cv::Vec3d vectorMulMatrix2Vector(const cv::Mat& matrix) {
    return {
        matrix.at<double>(2,1),
        matrix.at<double>(0,2),
        matrix.at<double>(1,0)
    };
}

cv::Affine3d eightPointAlgorithm(const std::vector<cv::Vec2d>& x_0_points, const std::vector<cv::Vec2d>& x_points) {
    cv::Mat matrixToSolve(8, 9, CV_64FC1);
    cv::Mat essentialMatrixCoefficients(1, 9, CV_64FC1);

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

    cv::Mat w, U, Vt, S;
    cv::SVD::compute(matrixToSolve, w, U, Vt, cv::SVD::FULL_UV);


    cv::Mat W = (cv::Mat_<double>(3,3) <<
        0, -1, 0,
        1,  0, 0,
        0,  0, 1
    );



    essentialMatrixCoefficients = Vt.row(8).clone();
    cv::Mat essentialMatrixEstimation = essentialMatrixCoefficients.reshape(1, 3).clone(); // usually incorrect coeficients in S matrix

    cv::SVD::compute(essentialMatrixEstimation, w, U, Vt, cv::SVD::FULL_UV);

    if (cv::determinant(U * Vt) < 0) {
        Vt = -Vt;
    }

    double s = (w.at<double>(0) + w.at<double>(1)) / 2.0;

    cv::Mat S_corrected = (cv::Mat_<double>(3,3) <<
        s, 0, 0,
        0, s, 0,
        0, 0, 0
    );

    cv::Mat t_x = U * W * S_corrected * U.t();
    cv::Mat R = U * W.inv() * Vt;





    if (cv::determinant(R) < 0) {
        R = -R;
        t_x = -t_x;
    }

    return cv::Affine3d(
        R,
        vectorMulMatrix2Vector(t_x)
    );
}

double scalarProduct(const cv::Vec3d& u, const cv::Vec3d& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

cv::Vec3d countProjection(const cv::Vec3d& u_point, const cv::Vec3d& v_point) //projection u on v, proj_u(v)
{
    double v_u_scalar = scalarProduct(v_point, u_point);
    double u_u_scalar = scalarProduct(u_point, u_point);
    return u_point * (v_u_scalar / u_u_scalar);
}

cv::Mat gramSchmidt(const cv::Mat &rotation_matrix) {
    cv::Mat R_orthogonal = rotation_matrix.clone();
    size_t n = rotation_matrix.rows;

    cv::Vec3d u0 = R_orthogonal.row(0);
    u0 /= cv::norm(u0);
    R_orthogonal.row(0) = cv::Mat(u0).t();

    // v_i - rows of the rotation matrix (unnormalized and not orthogonal)
    // u_i - rows of the new orthogonal matrix
    for (int i = 1; i < n; ++i) {
        cv::Vec3d u_i = rotation_matrix.row(i); // here its v_i
        for (int j = 0; j < i; ++j) {
            cv::Vec3d u_j = R_orthogonal.row(j);
            u_i -= countProjection(u_j, u_i);
        }
        u_i /= cv::norm(u_i);
        R_orthogonal.row(i) = cv::Mat(u_i).t();
    }
    return R_orthogonal;
}

std::vector<cv::Vec2d> world2Pixels(const std::vector<cv::Vec3d>& world_points_X,const cv::Matx33d& cameraMatrix,const  cv::Affine3d& P) {
    std::vector<cv::Vec2d> x_pixels;
    for (auto& p : world_points_X)
    {
        cv::Vec3d pixels = cameraMatrix * (P.rotation() * p + P.translation());
        x_pixels.emplace_back(pixels[0] / pixels[2], pixels[1] / pixels[2]);
    }

    return x_pixels;

}

void testPnP()
{
    cv::Vec3d rvec(0.1, -0.2, 0.3), t(1.1, -2.1, 3.);
    cv::Affine3d P(rvec, t);
    double fx = 300, fy = 300, cx = 320, cy = 320;
    cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

    std::vector<cv::Vec3d> X = { {3,2,5}, {10,-3,5.5}, {-3,1,4.5}, {8,2,4}, {-2,-3,4}, {8,3,10} };
    std::vector<cv::Vec2d> x;
    for (auto& p : X)
    {
        cv::Vec3d pixels = cameraMatrix * (P.rotation() * p + P.translation());
        x.push_back(cv::Vec2d(pixels[0] / pixels[2], pixels[1] / pixels[2]));
    }
    std::vector<double> distCoeffs(4, 0.);
    cv::Vec3d rvec1, tvec1;
    cv::solvePnP(X, x, cameraMatrix, distCoeffs, rvec1, tvec1, false, cv::SOLVEPNP_ITERATIVE);
    std::cout << rvec << t << "\n" << rvec1 << tvec1 << "\n";
}