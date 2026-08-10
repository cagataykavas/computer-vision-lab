from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    canvas = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (170, 120), (530, 380), (255, 255, 255), -1)
    cv2.putText(canvas, "CV LAB", (245, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    src = np.float32([[170, 120], [530, 120], [530, 380], [170, 380]])
    dst = np.float32([[120, 80], [590, 145], [545, 430], [145, 360]])
    h_true = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, h_true, (700, 500))

    h_est, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_est).reshape(-1, 2).astype(int)
    vis = warped.copy()
    for i in range(4):
        cv2.line(vis, tuple(projected[i]), tuple(projected[(i + 1) % 4]), (0, 255, 0), 3)

    error = float(np.mean(np.linalg.norm(projected.astype(float) - dst, axis=1)))
    cv2.putText(vis, f"mean corner error: {error:.3f}px", (20, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(str(out / "homography_demo.png"), vis)
    print({"mean_corner_error_px": error, "inliers": int(mask.sum()) if mask is not None else None})


if __name__ == "__main__":
    main()
