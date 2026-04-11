#!/usr/bin/env python3
"""
Shared utilities for image registration and feature matching.
Used by both assign_anchors.ipynb and calculate_coordinates.ipynb.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_image_file(path: Path) -> bool:
    """Check if path is a valid image file."""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_image_files(folder: Path) -> List[Path]:
    """List all image files in a folder, sorted."""
    files = [p for p in sorted(folder.iterdir()) if is_image_file(p)]
    if not files:
        raise FileNotFoundError(f"No image files found in: {folder}")
    return files


def load_image(path: Path) -> np.ndarray:
    """Load image from disk with OpenCV. Returns None if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: skipping unreadable file {path}")
        return None
    return img


def preprocess_for_matching(img_bgr: np.ndarray, clahe_clip: float = 2.0) -> np.ndarray:
    """Extract green channel and apply CLAHE for feature matching."""
    if img_bgr is None:
        raise ValueError("Failed to read image")
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    return clahe.apply(green)


def match_descriptors(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    ratio_thresh: float = 0.75,
) -> List[cv2.DMatch]:
    """Match SIFT descriptors using FLANN and Lowe's ratio test."""
    if des1 is None or des2 is None:
        return []
    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def estimate_homography(
    anchor_img_bgr: np.ndarray,
    test_img_bgr: np.ndarray,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 3.0,
    nfeatures: int = 4000,
    clahe_clip: float = 2.0,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """
    Estimate homography between anchor and test images using SIFT + RANSAC.
    Returns (H, stats) where H is the homography matrix and stats contains match info.
    """
    anchor = preprocess_for_matching(anchor_img_bgr, clahe_clip=clahe_clip)
    test = preprocess_for_matching(test_img_bgr, clahe_clip=clahe_clip)

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp_a, des_a = sift.detectAndCompute(anchor, None)
    kp_t, des_t = sift.detectAndCompute(test, None)

    stats = {
        "num_keypoints_anchor": len(kp_a),
        "num_keypoints_test": len(kp_t),
        "num_good_matches": 0,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
    }

    if des_a is None or des_t is None or len(des_a) < 4 or len(des_t) < 4:
        return None, stats

    good = match_descriptors(des_a, des_t, ratio_thresh=ratio_thresh)
    stats["num_good_matches"] = len(good)

    if len(good) < 4:
        return None, stats

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_t = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts_a, pts_t, cv2.RANSAC, ransac_thresh)
    if H is None or inlier_mask is None:
        return None, stats

    inliers = int(inlier_mask.ravel().sum())
    stats["num_inliers"] = inliers
    stats["inlier_ratio"] = inliers / max(len(good), 1)

    return H, stats


def transform_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography transformation to 2D points."""
    if H is None:
        raise ValueError("Homography is None; cannot transform points")
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return warped


def to_id(name: str, prefix: str) -> str:
    """Extract numeric ID from filename and format as prefix_NN."""
    stem = Path(name).stem
    m = re.search(r"(\d+)", stem)
    if m:
        return f"{prefix}_{int(m.group(1)):02d}"
    return stem


def load_anchor_points_csv(anchor_name: str, points_dir: Path) -> pd.DataFrame:
    """Load 100 sampled anchor points from CSV."""
    stem = Path(anchor_name).stem
    candidates = [
        points_dir / f"{stem}.csv",
        points_dir / f"{stem}_coordinates.csv",
        points_dir / f"{stem}_sampled_pixels.csv",
        points_dir / f"{stem}_sampled_points.csv",
        points_dir / f"{stem}_points.csv",
    ]

    if stem.startswith("anchor_"):
        suffix = stem.replace("anchor_", "")
        candidates.extend([
            points_dir / f"anchor_{suffix}.csv",
            points_dir / f"anchor_{suffix}_coordinates.csv",
            points_dir / f"anchor_{suffix}_sampled_pixels.csv",
        ])

    for p in candidates:
        if p.name.startswith("._"):
            continue
        if p.exists():
            df = pd.read_csv(p)
            cols = [c.lower().strip() for c in df.columns]
            if len(cols) < 2:
                raise ValueError(f"Anchor points csv has <2 columns: {p}")
            x_col = df.columns[0]
            y_col = df.columns[1]
            out = pd.DataFrame({
                "x": df[x_col].astype(float),
                "y": df[y_col].astype(float)
            })
            if len(out) != 100:
                print(f"Warning: expected 100 points but found {len(out)} in {p.name}")
            return out

    csvs = [p for p in sorted(points_dir.glob("*.csv")) if not p.name.startswith("._")]
    names = [c.name for c in csvs[:10]]
    raise FileNotFoundError(
        f"No anchor-point CSV found for {anchor_name}. Checked {len(candidates)} filename patterns in {points_dir}."
        f" Example csv files here: {names}"
    )
