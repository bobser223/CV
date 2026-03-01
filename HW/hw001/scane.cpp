//
// Created by Volodymyr Avvakumov on 28.01.2026.
//
#include "scane.h"





std::vector<Cuboid3d> createObjects() {
    std::vector<Cuboid3d> objects;

    Cuboid3d obj_apple(cv::Vec3d(1000, 0, 0), cv::Vec3d(612, 612, 608));

    obj_apple.textures[0] = obj_apple.textures[3] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple1.jpg");
    obj_apple.textures[1] = obj_apple.textures[4] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple2.jpg");
    obj_apple.textures[2] = obj_apple.textures[5] = cv::imread(PATH_TO_IMAGES + "apple//" + "apple3.jpg");



    objects.push_back(obj_apple);

    Cuboid3d obj_cherry(cv::Vec3d(500, 500, 0), cv::Vec3d(640, 560, 420)); // (L,W,H)

    obj_cherry.textures[0] = obj_cherry.textures[3] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry1.jpg");
    obj_cherry.textures[1] = obj_cherry.textures[4] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry2.jpg");
    obj_cherry.textures[2] = obj_cherry.textures[5] = cv::imread(PATH_TO_IMAGES + "cherry//" + "cherry3.jpg");
    objects.push_back(obj_cherry);

    Cuboid3d obj_pear(cv::Vec3d(1200, 0, 0), cv::Vec3d(704, 512, 448)); // (L,W,H)

    obj_pear.textures[0] = obj_pear.textures[3] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear1.jpg");
    obj_pear.textures[1] = obj_pear.textures[4] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear2.jpg");
    obj_pear.textures[2] = obj_pear.textures[5] = cv::imread(PATH_TO_IMAGES + "pear//" + "pear3.jpg");

    objects.push_back(obj_pear);
    return objects;
}

bool is_visible(const cv::Vec3d& image_point_NED,
    const cv::Vec3d& object_point_NED, const Cuboid3d::Id& outputObjId,
    const std::vector<Cuboid3d>& objects)
{
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

cv::Mat project2d(const cv::Affine3d& P,const std::vector<Cuboid3d>& objects, int imageWidth, int imageHeight, const cv::Matx33d& cameraMatrix)
{
    cv::Mat finalImage = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC3);
    const cv::Matx33d R = P.rotation();
    const cv::Vec3d t = P.translation();

    // camera position in NED (fi P: X_cam = R*X_world + t, than C_world = -R^T t)
    const cv::Vec3d C = -(cv::Vec3d)(R.t() * t);

    for (auto& obj : objects) {
        for (auto& texture : obj.textures){
            for (int faceId = 0; faceId < 6; faceId++) {
                const cv::Mat& tex = obj.textures[faceId];
                if (tex.empty()) {
                    throw std::runtime_error("Texture is empty");
                    continue;
                }

                for (int i = 0; i < tex.rows; ++i) {
                    for (int j = 0; j < tex.cols; ++j) {

                        const cv::Vec3d local = texel_to_local_point(faceId, i, j, tex.rows, tex.cols, obj.dimensions);
                        const auto Xopt = obj.local2NED(local);

                        if (!Xopt) continue;
                        const cv::Vec3d X = *Xopt; // point on the face in NED


                        const cv::Vec3d X_cam = R * X + t;
                        if (X_cam[2] <= 1e-9) continue;


                        const cv::Vec3d pix_h = cameraMatrix * X_cam;
                        const double px = pix_h[0] / pix_h[2];
                        const double py = pix_h[1] / pix_h[2];


                        if (px >= 0 && px < finalImage.cols && py >= 0 && py < finalImage.rows) {
                            if (is_visible(C, X, obj.id, objects)) {
                                finalImage.at<cv::Vec3b>(int(py), int(px)) = tex.at<cv::Vec3b>(i, j);
                            }
                        }




                    }
                }


            }}
    }


    return finalImage;
}











