//
// Created by Volodymyr Avvakumov on 28.01.2026.
//
#include "scane.h"
#include <algorithm>

#define DEFAULT_COLOR cv::Vec3b(255,255,255)

typedef size_t sz;


std::vector<Cuboid3d> createObjects() {
    std::vector<Cuboid3d> objects;

    Cuboid3d obj_apple(cv::Vec3d(10000, 0, 0), cv::Vec3d(612, 612, 608));

    obj_apple.textures[0] = obj_apple.textures[3] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple1.jpg");
    obj_apple.textures[1] = obj_apple.textures[4] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple2.jpg");
    obj_apple.textures[2] = obj_apple.textures[5] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple3.jpg");



    objects.push_back(obj_apple);

    Cuboid3d obj_cherry(cv::Vec3d(5000, -3000, 1500), cv::Vec3d(640, 560, 420)); // (L,W,H)

    obj_cherry.textures[0] = obj_cherry.textures[3] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry1.jpg");
    obj_cherry.textures[1] = obj_cherry.textures[4] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry2.jpg");
    obj_cherry.textures[2] = obj_cherry.textures[5] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry3.jpg");
    objects.push_back(obj_cherry);

    Cuboid3d obj_pear(cv::Vec3d(3000, 1600, -500), cv::Vec3d(704, 512, 448)); // (L,W,H)

    obj_pear.textures[0] = obj_pear.textures[3] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear1.jpg");
    obj_pear.textures[1] = obj_pear.textures[4] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear2.jpg");
    obj_pear.textures[2] = obj_pear.textures[5] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear3.jpg");

    objects.push_back(obj_pear);
    return objects;
}


bool is_visible(const cv::Vec3d& image_point_NED,
    const cv::Vec3d& object_point_NED, const Cuboid3d::Id& outputObjId,
    const std::vector<Cuboid3d>& objects)
{ // FIXME: algorithm is absolutely incorrect
    return true;
    for (auto& obj : objects) {
        // if (obj.id == outputObjId)
        //     continue;
        if (!obj.isSegmentCollisionFree(image_point_NED, object_point_NED)) return false;
    }
    return true;

}

static cv::Vec3d texel_to_local_point(
    int faceId, int i, int j,
    int rows, int cols,
    const cv::Vec3d& dim // (L, W, H)
) {
    const double L = dim[0], W = dim[1], H = dim[2];

    // normilized coords [0,1]
    const double u = (cols <= 1) ? 0.0 : double(j) / double(cols - 1);         // horisontal
    const double v = (rows <= 1) ? 0.0 : double(i) / double(rows - 1);         // vertival (down)
    const double up01 = 1.0 - v;                                               // up

    switch (faceId) {
        case 0: { // 1: e = 0, varies (n, up)
            const double n  = u * L;
            const double up = up01 * H;
            return {n, 0.0, up};
        }
        case 3: { // 4: e = W, varies (n, up)
            const double n  = u * L;
            const double up = up01 * H;
            return {n, W, up};
        }
        case 4: { // 5: n = 0, varies (e, up)
            const double e  = u * W;
            const double up = up01 * H;
            return {0.0, e, up};
        }
        case 1: { // 2: n = L, varies (e, up)
            const double e  = u * W;
            const double up = up01 * H;
            return {L, e, up};
        }
        case 2: { // 3: up = 0 (bottom), varies (n, e)
            const double n = u * L;
            const double e = v * W; //  v or 1-v, variates
            return {n, e, 0.0};
        }
        case 5: { // 6: up = H (top), varies (n, e)
            const double n = u * L;
            const double e = v * W;
            return {n, e, H};
        }
        default:
            return {0,0,0};
    }
}


static cv::Vec3d ray(const cv::Vec3d& C_w, const cv::Vec3d& d_w, double lambda ) {
    return C_w + lambda * d_w;
}


// static std::optional<sz> pointToFace(const cv::Vec3d& point_NED, const Cuboid3d& object) {
//
//     if (auto point_local_opt = object.NED2local(point_NED)) {
//         const cv::Vec3d& local = *point_local_opt;
//         const double length_obj = object.dimensions[0], width_obj = object.dimensions[1], height_obj = object.dimensions[2];
//         const double length_local = local[0], width_local = local[1], height_local = local[2];
//
//         if (length_local >= 0 && width_local >= 0 && height_local >= 0) {
//
//             if (length_local <= length_obj && height_local <= height_obj) { // 4, 1
//                 if (width_local == width_obj) {
//                     return {3};
//                 }
//                 if (width_local == 0) {
//                     return {0};
//                 }
//             }
//
//             if (width_local <= width_obj && height_local <= height_obj) { // 5, 2
//                 if (length_local == length_obj) {
//                     return {4};
//                 }
//                 if (length_local == 0) {
//                     return {1};
//                 }
//             }
//
//             if (length_local <= length_obj && width_local <= width_obj) { // 3, 6
//                 if (height_local == height_obj) {
//                     return {2};
//                 }
//                 if (height_local == 0) {
//                     return {5};
//                 }
//             }
//
//         }
//     }
//
//
//
//     return std::nullopt;
// }

static std::optional<sz> pointToFace(const cv::Vec3d& point_NED, const Cuboid3d& obj)
{
    auto localOpt = obj.NED2local(point_NED); // (n,e,up)
    if (!localOpt) return std::nullopt;

    const cv::Vec3d local = *localOpt;
    const double L = obj.dimensions[0], W = obj.dimensions[1], H = obj.dimensions[2];
    const double eps = 1e-6;

    if (std::abs(local[1] - 0.0) < eps) return 0; // e=0
    if (std::abs(local[1] - W)   < eps) return 3; // e=W
    if (std::abs(local[0] - 0.0) < eps) return 4; // n=0
    if (std::abs(local[0] - L)   < eps) return 1; // n=L
    if (std::abs(local[2] - 0.0) < eps) return 2; // up=0
    if (std::abs(local[2] - H)   < eps) return 5; // up=H

    return std::nullopt;
}

static void localToTexel(int faceId, const cv::Vec3d& local,
                         int rows, int cols, const cv::Vec3d& dim,
                         int& i, int& j)
{
    const double L = dim[0], W = dim[1], H = dim[2];

    double u = 0.0, v = 0.0; // u=horizontal [0..1], v=vertical [0..1] downwards

    switch (faceId) {
        case 0: // e=0, varies (n, up)
        case 3: // e=W
            u = (L > 0) ? (local[0] / L) : 0.0;
            v = (H > 0) ? (1.0 - local[2] / H) : 0.0;
            break;

        case 4: // n=0, varies (e, up)
        case 1: // n=L
            u = (W > 0) ? (local[1] / W) : 0.0;
            v = (H > 0) ? (1.0 - local[2] / H) : 0.0;
            break;

        case 2: // bottom up=0, varies (n,e)
        case 5: // top up=H
            u = (L > 0) ? (local[0] / L) : 0.0;
            v = (W > 0) ? (local[1] / W) : 0.0; // інколи треба 1-v, залежить від орієнтації картинки
            break;
    }

    j = (int)std::lround(u * (cols - 1));
    i = (int)std::lround(v * (rows - 1));
    i = std::clamp(i, 0, rows - 1);
    j = std::clamp(j, 0, cols - 1);
}

cv::Mat project2d(const cv::Affine3d& P,const std::vector<Cuboid3d>& objects, int imageWidth, int imageHeight, const cv::Matx33d& cameraMatrix)
{
    cv::Mat finalImage = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC3);
    const cv::Matx33d R = P.rotation();
    const cv::Vec3d t = P.translation();
    const cv::Matx33d Rt = R.t();
    const cv::Matx33d Kinv = cameraMatrix.inv();

    // camera position in NED (fi P: X_cam = R*X_world + t, than C_world = -R^T t)
    const cv::Vec3d C_w = -(cv::Vec3d)(R.t() * t);

    for (int a = 0; a < imageWidth; ++a) {
        for (int b= 0; b < imageHeight; ++b){
            auto d_c = Kinv * cv::Vec3d(a, b, 1);
            auto d_w = Rt * d_c;


            std::vector<cv::Vec3d> points(objects.size());
            std::vector<float> distances(objects.size(), std::numeric_limits<float>::max());
            std::vector<cv::Vec3d> intersection_points(objects.size());

            int i = 0;
            for (const auto& obj : objects) {
                if (auto point_opt = obj.getIntersectionPoint(C_w, d_w)) {
                    points[i] = *point_opt;
                    distances[i] = cv::norm(points[i] - C_w);
                    intersection_points[i] = points[i];
                }
                i++;
            }

            auto min_id_it = std::min_element(distances.begin(), distances.end());
            if (min_id_it != distances.end()) {
                sz min_id = min_id_it - distances.begin();
                if (distances[min_id] != std::numeric_limits<float>::max()) {
                    if (auto texture_idx_opt = pointToFace(intersection_points[min_id], objects[min_id])) {
                        if (auto local_px = objects[min_id].NED2local(intersection_points[min_id])) {
                            const cv::Mat& tex = objects[min_id].textures[*texture_idx_opt];
                            int ii, jj;
                            localToTexel((int)*texture_idx_opt, *local_px, tex.rows, tex.cols, objects[min_id].dimensions, ii, jj);
                            finalImage.at<cv::Vec3b>(b, a) = tex.at<cv::Vec3b>(ii, jj);
                        }

                    }
                }
            } else {
                finalImage.at<cv::Vec3b>(b, a) = DEFAULT_COLOR;
            }

        }
    }




    return finalImage;
}











