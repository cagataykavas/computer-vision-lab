from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


class HomographyEstimationError(ValueError):
    """Correspondences cannot support a trustworthy projective transform."""


@dataclass(frozen=True)
class HomographyDiagnostics:
    matrix: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    inlier_count: int
    inlier_ratio: float
    symmetric_rmse_px: float
    symmetric_p95_px: float
    condition_number: float

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix": self.matrix.tolist(),
            "inlier_mask": self.inlier_mask.tolist(),
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "symmetric_rmse_px": self.symmetric_rmse_px,
            "symmetric_p95_px": self.symmetric_p95_px,
            "condition_number": self.condition_number,
        }


def estimate_homography(
    source_points: NDArray[np.floating],
    destination_points: NDArray[np.floating],
    *,
    ransac_threshold_px: float = 3.0,
    confidence: float = 0.995,
    max_iterations: int = 5_000,
    minimum_inliers: int = 4,
    minimum_inlier_ratio: float = 0.5,
) -> HomographyDiagnostics:
    """Estimate a homography and fail closed when geometry or consensus is weak."""
    source = _validated_points(source_points, "source_points")
    destination = _validated_points(destination_points, "destination_points")
    if len(source) != len(destination):
        raise HomographyEstimationError("source and destination counts must match")
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise HomographyEstimationError("ransac_threshold_px must be positive and finite")
    if not 0 < confidence < 1:
        raise HomographyEstimationError("confidence must be between zero and one")
    if max_iterations <= 0:
        raise HomographyEstimationError("max_iterations must be positive")
    if not 4 <= minimum_inliers <= len(source):
        raise HomographyEstimationError("minimum_inliers must be between 4 and point count")
    if not 0 < minimum_inlier_ratio <= 1:
        raise HomographyEstimationError("minimum_inlier_ratio must be within (0, 1]")

    _reject_degenerate_layout(source, "source_points")
    _reject_degenerate_layout(destination, "destination_points")

    matrix, raw_mask = cv2.findHomography(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=max_iterations,
        confidence=confidence,
    )
    if matrix is None or raw_mask is None or not np.isfinite(matrix).all():
        raise HomographyEstimationError("OpenCV could not estimate a finite homography")
    if abs(float(matrix[2, 2])) < 1e-12:
        raise HomographyEstimationError("homography normalization is numerically unstable")

    matrix = np.asarray(matrix / matrix[2, 2], dtype=np.float64)
    inliers = raw_mask.reshape(-1).astype(bool)
    count = int(inliers.sum())
    ratio = count / len(inliers)
    if count < minimum_inliers:
        raise HomographyEstimationError(
            f"insufficient RANSAC consensus: {count} inliers, minimum is {minimum_inliers}"
        )
    if ratio < minimum_inlier_ratio:
        raise HomographyEstimationError(
            f"weak RANSAC consensus: ratio {ratio:.3f}, minimum is {minimum_inlier_ratio:.3f}"
        )

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise HomographyEstimationError("estimated homography is singular") from exc

    forward = _project(source[inliers], matrix)
    backward = _project(destination[inliers], inverse)
    forward_error = np.linalg.norm(forward - destination[inliers], axis=1)
    backward_error = np.linalg.norm(backward - source[inliers], axis=1)
    symmetric_error = np.sqrt((forward_error**2 + backward_error**2) / 2)

    return HomographyDiagnostics(
        matrix=matrix,
        inlier_mask=inliers,
        inlier_count=count,
        inlier_ratio=round(ratio, 6),
        symmetric_rmse_px=round(float(np.sqrt(np.mean(symmetric_error**2))), 6),
        symmetric_p95_px=round(float(np.percentile(symmetric_error, 95)), 6),
        condition_number=round(float(np.linalg.cond(matrix)), 6),
    )


def _validated_points(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise HomographyEstimationError(f"{name} must have shape (n, 2)")
    if len(points) < 4:
        raise HomographyEstimationError(f"{name} must contain at least four points")
    if not np.isfinite(points).all():
        raise HomographyEstimationError(f"{name} must contain only finite coordinates")
    if len(np.unique(points, axis=0)) < 4:
        raise HomographyEstimationError(f"{name} must contain at least four unique points")
    return points


def _reject_degenerate_layout(points: NDArray[np.float64], name: str) -> None:
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] <= 1e-12 or singular_values[1] / singular_values[0] < 1e-6:
        raise HomographyEstimationError(f"{name} are collinear or nearly collinear")


def _project(
    points: NDArray[np.float64],
    matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix)
    return projected.reshape(-1, 2)
