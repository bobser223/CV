//
// Created by Volodymyr Avvakumov on 17.03.2026.
//

#ifndef CODE_UTILS002_H
#define CODE_UTILS002_H


#include <opencv2/core.hpp>
#include <array>
#include <opencv2/core/affine.hpp>


cv::Affine3d eightPointAlgorithm(const std::array<cv::Vec3d, 8>& x_0_points, const std::array<cv::Vec3d, 8>& x_points);


#endif //CODE_UTILS002_H