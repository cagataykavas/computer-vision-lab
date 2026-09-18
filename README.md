# Computer Vision Lab

A compact portfolio of classical and modern computer-vision building blocks: feature matching, homography, motion analysis, tracking, segmentation and detection-oriented evaluation.

This repository complements the older project-specific repositories by exposing reusable utilities and small synthetic demos that can be run without private datasets.

## Included baseline

- synthetic perspective-warp demo
- fail-closed RANSAC homography estimation
- correspondence and collinearity validation
- inlier count/ratio, symmetric reprojection RMSE/P95 and matrix conditioning
- projected-corner visualization
- reusable geometry helpers
- links to older project-specific work such as HAZMAT detection, optical-flow grid analysis and tennis video processing

## Run

```bash
pip install -r requirements.txt
python demo_homography.py
```

The demo creates a synthetic scene, warps it with a known perspective transform, re-estimates the homography through the validated geometry boundary and saves a visualization under `artifacts/`.

## Robust geometry boundary

`estimate_homography()` wraps OpenCV's RANSAC estimator with behavior expected from a reusable vision component:

- rejects non-finite, duplicate, insufficient and nearly collinear correspondences;
- validates the RANSAC threshold, confidence, iteration and consensus policy;
- fails closed when the inlier count or inlier ratio is too low;
- normalizes the matrix and rejects singular solutions;
- reports forward/backward symmetric reprojection RMSE and p95;
- exposes the complete inlier mask and matrix condition number.

The condition number is diagnostic rather than a universal rejection threshold because coordinate scale materially affects it. Consumers should calibrate alert thresholds for their camera geometry.

## Verification

```bash
ruff check .
ruff format --check .
pytest -q
python demo_homography.py
```

Tests cover noisy recovery with 20% synthetic outliers, projection accuracy, weak-consensus rejection, degenerate layouts and invalid estimation policies. CI also publishes the generated visualization as evidence.

## Portfolio links

- `detectHazardsAndBarrels` — SIFT + RANSAC + HSV segmentation + lightweight tracking
- `rotationDetector` — Farnebäck optical flow + homography + 3x3 grid motion analysis
- `tennisGameProcessing` — court rectification + player/ball heuristics + top-down mapping
- `sleep-pose-estimation` — reserved for the original YOLOv8 capstone code once recovered

The emphasis is inspectable vision geometry, measurable failure semantics and evaluation rather than hiding everything behind a pretrained detector.
