//
// Created by Volodymyr Avvakumov on 17.03.2026.
//

#ifndef CODE_UTILS002_H
#define CODE_UTILS002_H


#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

std::vector<cv::Vec2d> normalizePixels(const std::vector<cv::Vec2d>& pixels, const cv::Matx33d& cameraMatrix);

cv::Vec3d vectorMulMatrix2Vector(const cv::Mat& matrix);

cv::Affine3d eightPointAlgorithm(const std::vector<cv::Vec2d>& x_0_points, const std::vector<cv::Vec2d>& x_points);

cv::Vec3d countProjection(const cv::Vec3d& u_point, const cv::Vec3d& v_point);

double scalarProduct(const cv::Vec3d& u, const cv::Vec3d& v);

cv::Mat gramSchmidt(const cv::Mat& rotation_matrix);

std::vector<cv::Vec2d> world2Pixels(const std::vector<cv::Vec3d>& world_points_X,const cv::Matx33d& cameraMatrix,const cv::Affine3d& P);

void testPnP();


#endif //CODE_UTILS002_H