//
// Created by Volodymyr Avvakumov on 28.01.2026.
//

#include "Cuboid3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::atomic<Cuboid3d::Id> Cuboid3d::next_id_{1};

Cuboid3d::Id Cuboid3d::make_id() {
    return next_id_.fetch_add(1, std::memory_order_relaxed);
}

std::optional<cv::Vec3d> Cuboid3d::local2NED(const cv::Vec3d& local) const {
    // local = (north, east, up) all >= 0
    if (local[0] < 0 || local[1] < 0 || local[2] < 0)
        return std::nullopt;

    // dimensions = (length_N, width_E, height_up) all >= 0
    if (local[0] > dimensions[0] || local[1] > dimensions[1] || local[2] > dimensions[2])
        return std::nullopt;

    // NED: +Z is Down, so Up is -Z
    return position + cv::Vec3d(local[0], local[1], -local[2]);
}

std::optional<cv::Vec3d> Cuboid3d::NED2local(const cv::Vec3d& global) const {
    const cv::Vec3d d = global - position;      // d in NED
    const cv::Vec3d local(d[0], d[1], -d[2]);   // convert to (N,E,Up)

    if (local[0] < 0 || local[1] < 0 || local[2] < 0) return std::nullopt;
    if (local[0] > dimensions[0] || local[1] > dimensions[1] || local[2] > dimensions[2]) return std::nullopt;

    return local; // <-- повертаємо local, не NED
}

bool Cuboid3d::isSegmentCollisionFree(const cv::Vec3d& point1, const cv::Vec3d& point2) const {
    // Build AABB in NED from reference corner "position" and "dimensions"
    const cv::Vec3d a = position;
    const cv::Vec3d b = position + cv::Vec3d(dimensions[0], dimensions[1], -dimensions[2]);

    const double minX = std::min(a[0], b[0]);
    const double maxX = std::max(a[0], b[0]);
    const double minY = std::min(a[1], b[1]);
    const double maxY = std::max(a[1], b[1]);
    const double minZ = std::min(a[2], b[2]);
    const double maxZ = std::max(a[2], b[2]);

    // Segment direction
    const cv::Vec3d d = point2 - point1;

    // Slab method for segment (t in [0,1])
    double tmin = 0.0;
    double tmax = 1.0;

    auto updateSlab = [&](double p, double dir, double mn, double mx) -> bool {
        constexpr double eps = 1e-12;

        if (std::abs(dir) < eps) {
            // Segment is parallel to slab; must be within slab to intersect
            return (p >= mn && p <= mx);
        }

        double t1 = (mn - p) / dir;
        double t2 = (mx - p) / dir;
        if (t1 > t2) std::swap(t1, t2);

        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);

        return tmin <= tmax;
    };

    if (!updateSlab(point1[0], d[0], minX, maxX)) return true;  // no intersection => collision-free
    if (!updateSlab(point1[1], d[1], minY, maxY)) return true;
    if (!updateSlab(point1[2], d[2], minZ, maxZ)) return true;

    // If we got here, segment intersects the box for some t in [0,1]
    return false;
}



std::optional<cv::Vec3d> Cuboid3d::getIntersectionPoint(const cv::Vec3d& C_w,
                                                        const cv::Vec3d& d_w) const
{
    // AABB of the cuboid in NED
    const cv::Vec3d a = position;
    const cv::Vec3d b = position + cv::Vec3d(dimensions[0], dimensions[1], -dimensions[2]);

    const double minX = std::min(a[0], b[0]);
    const double maxX = std::max(a[0], b[0]);
    const double minY = std::min(a[1], b[1]);
    const double maxY = std::max(a[1], b[1]);
    const double minZ = std::min(a[2], b[2]);
    const double maxZ = std::max(a[2], b[2]);

    constexpr double eps = 1e-12;

    double tmin = -std::numeric_limits<double>::infinity();
    double tmax =  std::numeric_limits<double>::infinity();

    auto slab = [&](double o, double d, double mn, double mx) -> bool {
        if (std::abs(d) < eps) {
            // Ray parallel to slab: must be inside slab
            return (o >= mn && o <= mx);
        }

        double t1 = (mn - o) / d;
        double t2 = (mx - o) / d;
        if (t1 > t2) std::swap(t1, t2);

        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        return tmin <= tmax;
    };

    if (!slab(C_w[0], d_w[0], minX, maxX)) return std::nullopt;
    if (!slab(C_w[1], d_w[1], minY, maxY)) return std::nullopt;
    if (!slab(C_w[2], d_w[2], minZ, maxZ)) return std::nullopt;

    // If the whole box is "behind" the ray origin
    if (tmax < 0.0) return std::nullopt;

    // Pick the first intersection in front of the origin:
    // - outside box: tmin >= 0 => entry
    // - inside box:  tmin < 0  => exit (tmax)
    double thit = (tmin >= 0.0) ? tmin : tmax;

    if (thit < 0.0) return std::nullopt;

    return C_w + thit * d_w;
}