# Computer Vision Lab

A compact portfolio of classical and modern computer-vision building blocks: feature matching, homography, motion analysis, tracking, segmentation and detection-oriented evaluation.

This repository complements the older project-specific repositories by exposing reusable utilities and small synthetic demos that can be run without private datasets.

## Included baseline

- synthetic perspective-warp demo
- homography estimation
- projected-corner visualization
- reusable geometry helpers
- links to older project-specific work such as HAZMAT detection, optical-flow grid analysis and tennis video processing

## Run

```bash
pip install -r requirements.txt
python demo_homography.py
```

The demo creates a synthetic scene, warps it with a known perspective transform, re-estimates the homography from correspondences and saves a visualization under `artifacts/`.

## Portfolio links

- `detectHazardsAndBarrels` — SIFT + RANSAC + HSV segmentation + lightweight tracking
- `rotationDetector` — Farnebäck optical flow + homography + 3x3 grid motion analysis
- `tennisGameProcessing` — court rectification + player/ball heuristics + top-down mapping
- `sleep-pose-estimation` — reserved for the original YOLOv8 capstone code once recovered

The emphasis is inspectable vision geometry and evaluation rather than hiding everything behind a pretrained detector.
