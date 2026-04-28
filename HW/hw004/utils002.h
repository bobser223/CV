//
// Created by Volodymyr Avvakumov on 20.04.2026.
//

#ifndef CODE_UTILS002_H
#define CODE_UTILS002_H


#include <opencv2/core/affine.hpp>
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


#endif //CODE_UTILS002_H
