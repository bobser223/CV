import cv2
import numpy as np


def generate_synthetic_pnp(
        out_path="synthetic_pnp.yaml",
        n_points=100,
        outlier_ratio=0.30,
        noise_std_px=1.5,
        image_width=1280,
        image_height=720,
        seed=123
):
    rng = np.random.default_rng(seed)

    # ---------------- camera intrinsics ----------------
    fx, fy = 800.0, 800.0
    cx, cy = image_width / 2.0, image_height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    # ---------------- ground truth pose ----------------
    # world -> camera
    rvec_gt = np.array([[0.25], [-0.18], [0.10]], dtype=np.float64)
    tvec_gt = np.array([[0.4], [-0.2], [1.2]], dtype=np.float64)

    R_gt, _ = cv2.Rodrigues(rvec_gt)

    # ---------------- generate 3D points ----------------
    # Зручно генерувати точки спочатку в camera-coordinates,
    # щоб вони точно були перед камерою, а потім перевести в world.
    Xc = np.empty((n_points, 3), dtype=np.float64)
    Xc[:, 0] = rng.uniform(-2.0, 2.0, size=n_points)   # x
    Xc[:, 1] = rng.uniform(-1.5, 1.5, size=n_points)   # y
    Xc[:, 2] = rng.uniform(4.0, 9.0, size=n_points)    # z > 0

    # Xc = R * Xw + t  =>  Xw = R^T * (Xc - t)
    points3d = ((R_gt.T @ (Xc.T - tvec_gt)).T).astype(np.float64)

    # ---------------- perfect projection ----------------
    pixels2d, _ = cv2.projectPoints(points3d, rvec_gt, tvec_gt, K, dist_coeffs)
    pixels2d = pixels2d.reshape(-1, 2)

    # ---------------- add Gaussian noise ----------------
    pixels2d += rng.normal(0.0, noise_std_px, size=pixels2d.shape)

    # ---------------- inject outliers ----------------
    n_outliers = int(round(n_points * outlier_ratio))
    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)

    inlier_mask_gt = np.ones((n_points, 1), dtype=np.uint8)
    inlier_mask_gt[outlier_idx] = 0

    # Для outliers просто підміняємо 2D координати випадковими пікселями
    pixels2d[outlier_idx, 0] = rng.uniform(0, image_width - 1, size=n_outliers)
    pixels2d[outlier_idx, 1] = rng.uniform(0, image_height - 1, size=n_outliers)

    # ---------------- save to YAML ----------------
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_width))
    fs.write("image_height", int(image_height))
    fs.write("cameraMatrix", K)
    fs.write("distCoeffs", dist_coeffs)
    fs.write("rvec_gt", rvec_gt)
    fs.write("tvec_gt", tvec_gt)
    fs.write("points3d", points3d)
    fs.write("pixels2d", pixels2d)
    fs.write("inlier_mask_gt", inlier_mask_gt)
    fs.release()

    print(f"saved: {out_path}")
    print(f"n_points={n_points}")
    print(f"noise_std_px={noise_std_px}")
    print(f"outlier_ratio={outlier_ratio:.2f}")
    print(f"n_outliers={n_outliers}")


if __name__ == "__main__":

    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp.yaml",
        n_points=120,
        outlier_ratio=0.35,
        noise_std_px=1.5,
        image_width=1280,
        image_height=720,
        seed=123
    )

    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp_1.yaml",
        n_points=120,
        outlier_ratio=0.0,
        noise_std_px=0.1,
        image_width=1280,
        image_height=720,
        seed=123
    )

    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp_2.yaml",
        n_points=120,
        outlier_ratio=0.0,
        noise_std_px= 1.0,
        image_width=1280,
        image_height=720,
        seed=123
    )


    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp_3.yaml",
        n_points=120,
        outlier_ratio=0.2,
        noise_std_px= 1.5,
        image_width=1280,
        image_height=720,
        seed=123
    )

    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp_4.yaml",
        n_points=120,
        outlier_ratio=0.4,
        noise_std_px= 2.0,
        image_width=1280,
        image_height=720,
        seed=123
    )


    generate_synthetic_pnp(
        out_path="../../data/hw003/synthetic_pnp_5.yaml",
        n_points=120,
        outlier_ratio=0.5,
        noise_std_px= 3.0,
        image_width=1280,
        image_height=720,
        seed=123
    )




