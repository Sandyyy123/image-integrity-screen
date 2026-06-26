"""Duplication detection for scientific figures.

Two screens:
  1. copy_move(image)        - find a region duplicated WITHIN one panel
                               (the classic "same blot pasted twice" tell).
  2. cross_duplication(a, b) - find a region shared ACROSS two panels/figures
                               (e.g. a control band reused under a new label).

Both use ORB keypoints + a ratio-tested brute-force matcher, then keep only
geometrically consistent match clusters. ORB is rotation/scale aware, so it
catches duplications that survive flips, small rotations, and rescaling.

Screening aid only: every cluster is shown to a human, who confirms or dismisses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DupResult:
    check: str
    matches: int               # geometrically consistent matched keypoints
    flagged: bool
    note: str
    points: list = field(default_factory=list)  # sample matched coords for overlay

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "consistent_matches": self.matches,
            "flagged": self.flagged,
            "note": self.note,
        }


def _gray(path_or_img):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path_or_img)
        return img
    if path_or_img.ndim == 3:
        return cv2.cvtColor(path_or_img, cv2.COLOR_BGR2GRAY)
    return path_or_img


def _orb_match(g1, g2, ratio=0.75, min_spatial_sep=8, same_image=False):
    orb = cv2.ORB_create(nfeatures=4000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return [], k1 or [], k2 or []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []

    if same_image:
        # Self-matching: the nearest neighbour of every keypoint is itself, so we
        # ask for k=3 and drop the trivial self-pair before the ratio test. The
        # surviving best match is the candidate duplicate elsewhere in the panel.
        raw = bf.knnMatch(d1, d1, k=3)
        for pair in raw:
            cand = [m for m in pair if m.trainIdx != m.queryIdx]
            if len(cand) < 2:
                continue
            m, n = cand[0], cand[1]
            if m.distance < ratio * n.distance:
                p1 = np.array(k1[m.queryIdx].pt)
                p2 = np.array(k1[m.trainIdx].pt)
                if np.linalg.norm(p1 - p2) >= min_spatial_sep:
                    good.append((m, p1, p2))
        return good, k1, k1

    raw = bf.knnMatch(d1, d2, k=2)
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            p1 = np.array(k1[m.queryIdx].pt)
            p2 = np.array(k2[m.trainIdx].pt)
            if np.linalg.norm(p1 - p2) >= min_spatial_sep:
                good.append((m, p1, p2))
    return good, k1, k2


def _geometric_filter(good, reproj=5.0):
    """Keep matches consistent with a single homography (a real pasted region)."""
    if len(good) < 8:
        return [], None
    src = np.float32([p1 for _, p1, _ in good]).reshape(-1, 1, 2)
    dst = np.float32([p2 for _, _, p2 in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    if mask is None:
        return [], None
    inliers = [g for g, keep in zip(good, mask.ravel()) if keep]
    return inliers, H


def copy_move(path_or_img, flag_at=12) -> DupResult:
    g = _gray(path_or_img)
    good, _, _ = _orb_match(g, g, same_image=True)
    inliers, _ = _geometric_filter(good)
    n = len(inliers)
    pts = [[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])] for _, p1, p2 in inliers[:30]]
    flagged = n >= flag_at
    note = (f"{n} geometrically consistent self-matches: region likely duplicated "
            f"within the panel." if flagged else
            f"{n} self-matches; below flag threshold.")
    return DupResult("copy_move_within_panel", n, flagged, note, pts)


def cross_duplication(path_a, path_b, flag_at=15) -> DupResult:
    g1, g2 = _gray(path_a), _gray(path_b)
    good, _, _ = _orb_match(g1, g2, min_spatial_sep=0)
    inliers, _ = _geometric_filter(good)
    n = len(inliers)
    pts = [[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])] for _, p1, p2 in inliers[:30]]
    flagged = n >= flag_at
    note = (f"{n} consistent matches across the two panels: a shared/reused region "
            f"is likely." if flagged else
            f"{n} cross matches; below flag threshold.")
    return DupResult("duplication_across_panels", n, flagged, note, pts)
