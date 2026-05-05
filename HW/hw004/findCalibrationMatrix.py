import cv2
import numpy as np
import glob
import os


# ============================================================
# НАЛАШТУЙ ЦЕ ПІД СВОЮ ДОШКУ
# ============================================================

# Кількість ВНУТРІШНІХ кутів: columns, rows
CHECKERBOARD_SIZE = (9, 6)

# Розмір однієї клітинки в міліметрах.
# Якщо тобі потрібна тільки camera matrix K, масштаб не критичний.
# Але краще поставити реальний розмір, наприклад 25.0 мм.
SQUARE_SIZE = 26.0

# Папка з jpg-фото
IMAGE_FOLDER = "../../data/hw004/chess_board_iphone_17_pro_main/jpg"

# ============================================================


def main():
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.JPG")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.JPEG"))
    )

    if not image_paths:
        raise RuntimeError(f"No images found in folder: {IMAGE_FOLDER}")

    print(f"Found {len(image_paths)} images")

    # Критерій для sub-pixel уточнення кутів
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # 3D-точки шахматної дошки в її локальній системі координат
    # Всі точки лежать на площині z = 0
    objp = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD_SIZE[0],
        0:CHECKERBOARD_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE

    object_points = []  # 3D точки в координатах дошки
    image_points = []   # 2D точки на зображенні

    image_size = None
    used_images = []

    os.makedirs("debug_corners", exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"[SKIP] Cannot read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        current_image_size = gray.shape[::-1]  # width, height

        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            print(f"[SKIP] Different image size: {path}")
            print(f"       expected {image_size}, got {current_image_size}")
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                  + cv2.CALIB_CB_NORMALIZE_IMAGE
                  + cv2.CALIB_CB_FAST_CHECK
        )

        if not found:
            print(f"[FAIL] Corners not found: {path}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        object_points.append(objp.copy())
        image_points.append(corners_refined)
        used_images.append(path)

        debug_img = img.copy()
        cv2.drawChessboardCorners(
            debug_img,
            CHECKERBOARD_SIZE,
            corners_refined,
            found
        )

        debug_name = os.path.basename(path)
        cv2.imwrite(os.path.join("debug_corners", debug_name), debug_img)

        print(f"[OK] {path}")

    print()
    print(f"Used images: {len(object_points)} / {len(image_paths)}")

    if len(object_points) < 5:
        raise RuntimeError(
            "Too few valid images for calibration. "
            "Need at least 5, better 15-30."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    print()
    print("========== CALIBRATION RESULT ==========")
    print()
    print("Image size:")
    print(image_size)

    print()
    print("Reprojection error:")
    print(ret)

    print()
    print("Camera matrix K:")
    print(camera_matrix)

    print()
    print("Distortion coefficients:")
    print(dist_coeffs)

    # Збереження в файл
    np.savez(
        "camera_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error=ret,
        checkerboard_size=CHECKERBOARD_SIZE,
        square_size=SQUARE_SIZE
    )

    print()
    print("Saved to camera_calibration.npz")

    # Також збережемо у YAML-подібний OpenCV файл
    fs = cv2.FileStorage("camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("reprojection_error", ret)
    fs.release()

    print("Saved to camera_calibration.yaml")

    print()
    print("Debug images with detected corners saved to debug_corners/")


if __name__ == "__main__":
    main()