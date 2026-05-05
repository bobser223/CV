import cv2
import numpy as np


IMAGE_PATH = "../../data/hw004/sheep_iphone_17_pro_main/jpg/IMG_1529.jpg"


def nothing(x):
    pass


def main():
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    window_name = "Good Features To Track"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("maxCorners", window_name, 200, 2000, nothing)
    cv2.createTrackbar("quality x1000", window_name, 10, 100, nothing)
    cv2.createTrackbar("minDistance", window_name, 10, 100, nothing)
    cv2.createTrackbar("blockSize", window_name, 3, 30, nothing)
    cv2.createTrackbar("useHarris", window_name, 0, 1, nothing)
    cv2.createTrackbar("k x1000", window_name, 40, 200, nothing)

    while True:
        max_corners = cv2.getTrackbarPos("maxCorners", window_name)
        quality_raw = cv2.getTrackbarPos("quality x1000", window_name)
        min_distance = cv2.getTrackbarPos("minDistance", window_name)
        block_size = cv2.getTrackbarPos("blockSize", window_name)
        use_harris = cv2.getTrackbarPos("useHarris", window_name)
        k_raw = cv2.getTrackbarPos("k x1000", window_name)

        # Захист від некоректних значень
        max_corners = max(max_corners, 1)
        quality_level = max(quality_raw / 1000.0, 0.001)
        min_distance = max(min_distance, 1)

        # blockSize має бути >= 2
        block_size = max(block_size, 2)

        # Бажано непарний blockSize
        if block_size % 2 == 0:
            block_size += 1

        k = k_raw / 1000.0

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            useHarrisDetector=bool(use_harris),
            k=k
        )

        vis = img.copy()

        if corners is not None:
            corners = corners.astype(np.int32)

            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

            count_text = f"corners: {len(corners)}"
        else:
            count_text = "corners: 0"

        cv2.putText(
            vis,
            count_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()