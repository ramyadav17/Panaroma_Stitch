"""Panorama stitching functions extracted from Panorama_Stitching_Project.ipynb."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


def resize_for_processing(image: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    """Limit very large uploads so interactive stitching remains responsive."""
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def detect_features(image: np.ndarray, method: str = "sift", n_features: int = 2000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=n_features)
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError("method must be 'sift' or 'orb'")
    return detector.detectAndCompute(gray, None)


def match_features(des1: np.ndarray, des2: np.ndarray, method: str = "sift", ratio: float = 0.75):
    if des1 is None or des2 is None:
        return []
    norm = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING
    raw_matches = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
    return [m for pair in raw_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Direct Linear Transform implementation from the notebook."""
    rows = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        rows.extend(([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp],
                     [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homography = vt[-1].reshape(3, 3)
    if abs(homography[2, 2]) < 1e-10:
        raise np.linalg.LinAlgError("Degenerate homography")
    return homography / homography[2, 2]


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2, None]


def ransac_homography(src_pts: np.ndarray, dst_pts: np.ndarray, max_iter: int = 1200,
                      threshold: float = 5.0, seed: int = 42):
    """Robust homography estimator, preserving the notebook's custom RANSAC."""
    if len(src_pts) < 4:
        raise RuntimeError("At least four matches are required")
    best_inliers = np.array([], dtype=int)
    best_homography = None
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        indices = rng.choice(len(src_pts), 4, replace=False)
        try:
            candidate = compute_homography(src_pts[indices], dst_pts[indices])
            errors = np.linalg.norm(apply_homography(candidate, src_pts) - dst_pts, axis=1)
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_homography = inliers, candidate
    if best_homography is None or len(best_inliers) < 4:
        raise RuntimeError("RANSAC could not find a stable alignment. Try images with more overlap.")
    return compute_homography(src_pts[best_inliers], dst_pts[best_inliers]), best_inliers


def warp_image(image: np.ndarray, homography: np.ndarray, out_size: tuple[int, int]):
    """Inverse-mapped, bilinear image warp from the notebook."""
    out_width, out_height = out_size
    inverse = np.linalg.inv(homography)
    xs, ys = np.meshgrid(np.arange(out_width), np.arange(out_height))
    destination = np.column_stack((xs.ravel(), ys.ravel(), np.ones(xs.size)))
    source = (inverse @ destination.T).T
    source /= source[:, 2:3]
    source_x, source_y = source[:, 0].reshape(out_height, out_width), source[:, 1].reshape(out_height, out_width)
    height, width = image.shape[:2]
    valid = (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    x0 = np.clip(np.floor(source_x).astype(int), 0, width - 2)
    y0 = np.clip(np.floor(source_y).astype(int), 0, height - 2)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = source_x - x0, source_y - y0
    result = np.zeros((out_height, out_width, 3), dtype=np.float32)
    for channel in range(3):
        ia, ib = image[y0, x0, channel], image[y0, x1, channel]
        ic, id_ = image[y1, x0, channel], image[y1, x1, channel]
        values = ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy
        result[..., channel] = np.where(valid, values, 0)
    return result, valid.astype(np.float32)


def feather_blend(base: np.ndarray, base_mask: np.ndarray, warped: np.ndarray, warped_mask: np.ndarray):
    base_distance = cv2.distanceTransform((base_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    warped_distance = cv2.distanceTransform((warped_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = (warped_distance / np.maximum(base_distance + warped_distance, 1e-6))[..., None]
    overlap = (base_mask > 0) & (warped_mask > 0)
    result = np.zeros_like(base)
    result[overlap] = warped[overlap] * alpha[overlap] + base[overlap] * (1 - alpha[overlap])
    result[(warped_mask > 0) & (base_mask == 0)] = warped[(warped_mask > 0) & (base_mask == 0)]
    result[(base_mask > 0) & (warped_mask == 0)] = base[(base_mask > 0) & (warped_mask == 0)]
    return result


def stitch_pair(base_image: np.ndarray, new_image: np.ndarray, method: str = "sift", ratio: float = 0.75,
                ransac_threshold: float = 5.0, blend: str = "feather", canvas_pad: int = 300):
    stats: dict[str, Any] = {"method": method}
    base_padded = cv2.copyMakeBorder(base_image, canvas_pad, canvas_pad, canvas_pad, canvas_pad, cv2.BORDER_CONSTANT)
    base_mask = (base_padded.sum(axis=2) > 0).astype(np.float32)
    started = time.perf_counter()
    new_keypoints, new_descriptors = detect_features(new_image, method)
    base_keypoints, base_descriptors = detect_features(base_image, method)
    stats["feature_time_s"] = time.perf_counter() - started
    good_matches = match_features(new_descriptors, base_descriptors, method, ratio)
    if len(good_matches) < 4:
        raise RuntimeError(f"Only {len(good_matches)} good matches were found. Use overlapping photos.")
    source = np.float32([new_keypoints[m.queryIdx].pt for m in good_matches])
    destination = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]) + canvas_pad
    started = time.perf_counter()
    homography, inliers = ransac_homography(source, destination, threshold=ransac_threshold)
    stats["alignment_time_s"] = time.perf_counter() - started
    reprojection = apply_homography(homography, source[inliers])
    stats.update(num_matches=len(good_matches), num_inliers=len(inliers),
                 inlier_ratio=round(len(inliers) / len(good_matches), 3),
                 reprojection_error_px=round(float(np.mean(np.linalg.norm(reprojection - destination[inliers], axis=1))), 3))
    warped, warped_mask = warp_image(new_image.astype(np.float32), homography, (base_padded.shape[1], base_padded.shape[0]))
    if blend == "feather":
        result = feather_blend(base_padded.astype(np.float32), base_mask, warped, warped_mask)
    else:
        result = base_padded.astype(np.float32)
        result[warped_mask > 0] = warped[warped_mask > 0]
    return result.astype(np.uint8), stats


def crop_to_content(image: np.ndarray) -> np.ndarray:
    mask = image.sum(axis=2) > 0
    rows, cols = np.where(mask)
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1] if len(rows) else image


def stitch_images(images: list[np.ndarray], **settings):
    if len(images) < 2:
        raise ValueError("Upload at least two images")
    canvas = images[len(images) // 2]
    diagnostics = []
    for index, image in enumerate(images):
        if index == len(images) // 2:
            continue
        canvas, stats = stitch_pair(canvas, image, **settings)
        stats["step"] = len(diagnostics) + 1
        diagnostics.append(stats)
    return crop_to_content(canvas), diagnostics
