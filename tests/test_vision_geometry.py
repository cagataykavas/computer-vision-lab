from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_geometry import HomographyEstimationError, estimate_homography


def _synthetic_correspondences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    source = rng.uniform([0, 0], [640, 480], size=(60, 2)).astype(np.float64)
    expected = np.array(
        [[1.05, 0.08, 24.0], [-0.04, 0.97, 18.0], [0.0002, -0.00015, 1.0]],
        dtype=np.float64,
    )
    destination = cv2.perspectiveTransform(source.reshape(-1, 1, 2), expected).reshape(-1, 2)
    destination += rng.normal(0, 0.35, size=destination.shape)
    destination[-12:] = rng.uniform([0, 0], [640, 480], size=(12, 2))
    return source, destination, expected


def test_robust_estimator_recovers_transform_and_reports_quality() -> None:
    source, destination, expected = _synthetic_correspondences()

    result = estimate_homography(
        source,
        destination,
        ransac_threshold_px=2.0,
        minimum_inlier_ratio=0.7,
    )

    assert 45 <= result.inlier_count <= 50
    assert result.inlier_ratio >= 0.75
    assert result.symmetric_rmse_px < 1.0
    assert result.symmetric_p95_px < 1.5
    assert np.isfinite(result.condition_number)

    check_points = np.array([[10, 20], [300, 200], [620, 450]], dtype=np.float64)
    estimated = cv2.perspectiveTransform(
        check_points.reshape(-1, 1, 2), result.matrix
    ).reshape(-1, 2)
    truth = cv2.perspectiveTransform(check_points.reshape(-1, 1, 2), expected).reshape(-1, 2)
    assert float(np.max(np.linalg.norm(estimated - truth, axis=1))) < 1.0
    assert result.to_dict()["inlier_count"] == result.inlier_count


@pytest.mark.parametrize(
    "points",
    [
        np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float),
        np.array([[0, 0], [0, 0], [1, 1], [2, 3]], dtype=float),
        np.array([[0, 0], [1, 0], [1, 1]], dtype=float),
        np.array([[0, 0], [1, 0], [1, np.nan], [0, 1]], dtype=float),
    ],
)
def test_invalid_or_degenerate_geometry_is_rejected(points: np.ndarray) -> None:
    with pytest.raises(HomographyEstimationError):
        estimate_homography(points, points)


def test_weak_outlier_consensus_fails_closed() -> None:
    source, destination, _ = _synthetic_correspondences()

    with pytest.raises(HomographyEstimationError, match="weak RANSAC consensus"):
        estimate_homography(
            source,
            destination,
            ransac_threshold_px=2.0,
            minimum_inlier_ratio=0.9,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ransac_threshold_px": 0},
        {"confidence": 1.0},
        {"max_iterations": 0},
        {"minimum_inliers": 3},
        {"minimum_inlier_ratio": 0},
    ],
)
def test_invalid_estimation_policy_is_rejected(kwargs: dict[str, float]) -> None:
    source, destination, _ = _synthetic_correspondences()

    with pytest.raises(HomographyEstimationError):
        estimate_homography(source, destination, **kwargs)
