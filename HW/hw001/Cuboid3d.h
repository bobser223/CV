//
// Created by Volodymyr Avvakumov on 28.01.2026.
//

#ifndef CODE_OBJ3D_H
#define CODE_OBJ3D_H

#include <opencv2/core.hpp>
#include <optional>
#include <atomic>
#include <cstdint>
#include <array>


class Cuboid3d {
public:
    using Id = std::uint64_t;

    Id id;

    cv::Vec3d position;    // reference corner in NED: (N0, E0, D0) at local (0,0,0)
    cv::Vec3d dimensions;  // (length_N, width_E, height_Up), all >= 0

    std::array<cv::Mat, 6> textures;

    Cuboid3d()
    : id(make_id()),
    position(0, 0, 0),
      dimensions(0, 0, 0)
    {}

    Cuboid3d(const cv::Vec3d& pos)
    : id(make_id()),
    position(pos),
      dimensions(0,0,0)
    {}

    Cuboid3d(const cv::Vec3d& pos , const cv::Vec3d& dim)
        :id(make_id()),
        position(pos),
        dimensions(dim)
    {}

    ~Cuboid3d() = default;

    std::optional<cv::Mat> getFrontTexture() const {
        if (textures[0].empty()) return std::nullopt;
        return textures[0];
    }

    std::optional<cv::Vec3d> local2NED(const cv::Vec3d& local) const;

    std::optional<cv::Vec3d> NED2local(const cv::Vec3d& global) const;

    bool isSegmentCollisionFree(const cv::Vec3d& point1, const cv::Vec3d& point2) const;

private:
    static std::atomic<Id> next_id_;
    static Id make_id();

};


#endif //CODE_OBJ3D_H