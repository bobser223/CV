//
// Created by Volodymyr Avvakumov on 28.01.2026.
//

#ifndef CODE_SCANE_H
#define CODE_SCANE_H

#include "Cuboid3d.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <filesystem>

static const std::string PATH_TO_IMAGES =
    (std::filesystem::path(__FILE__).parent_path() / "../../data/hw001").lexically_normal().string()
    + "/";

std::vector<Cuboid3d> createObjects();

bool is_visible(const cv::Vec3d& image_point_NED,
    const cv::Vec3d& object_point_NED, const Cuboid3d::Id& outputObjId,
    const std::vector<Cuboid3d>& objects);

static cv::Vec3d texel_to_local_point(
    int faceId, int i, int j,
    int rows, int cols,
    const cv::Vec3d& dim // (L, W, H)
);


cv::Mat project2d(const cv::Affine3d& P,const std::vector<Cuboid3d>& objects, int imageWidth, int imageHeight, const cv::Matx33d& cameraMatrix);


#endif //CODE_SCANE_H