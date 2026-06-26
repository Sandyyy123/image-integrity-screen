"""Error-Level Analysis (ELA) for scientific-figure manipulation screening.

ELA re-saves an image at a known JPEG quality and amplifies the per-pixel
difference against the original. Regions that have been pasted, cloned, or
locally edited tend to have a different recompression response than the
surrounding native pixels, so they light up in the ELA map. A high-variance,
sharp-edged bright block is a manipulation flag worth a manual look + a request
for the original uncropped file.

This is a screening aid, not proof. Every flag must be confirmed by a human.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops


@dataclass
class ELAResult:
    max_diff: float            # brightest recompression difference (0-255)
    mean_diff: float           # mean difference across the image
    hotspot_ratio: float       # fraction of pixels well above the mean (suspicious area)
    flagged: bool              # screening verdict
    note: str

    def as_dict(self) -> dict:
        return {
            "check": "error_level_analysis",
            "max_diff": round(self.max_diff, 2),
            "mean_diff": round(self.mean_diff, 2),
            "hotspot_ratio": round(self.hotspot_ratio, 4),
            "flagged": self.flagged,
            "note": self.note,
        }


def run_ela(path: str, quality: int = 90, hotspot_sigma: float = 4.0,
            save_map: str | None = None) -> ELAResult:
    """Run ELA on an image file. Optionally write the amplified ELA map to disk."""
    original = Image.open(path).convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    ela = ImageChops.difference(original, recompressed)
    arr = np.asarray(ela).astype(np.float32)
    per_pixel = arr.max(axis=2)  # strongest channel difference per pixel

    max_diff = float(per_pixel.max())
    mean_diff = float(per_pixel.mean())
    std_diff = float(per_pixel.std()) or 1.0

    # pixels whose recompression error sits far above the image's own mean
    threshold = mean_diff + hotspot_sigma * std_diff
    hotspot_ratio = float((per_pixel > threshold).mean())

    # A localized, bright, sharp-edged cluster is the signature we care about.
    flagged = max_diff >= 60 and hotspot_ratio >= 0.002 and (max_diff / mean_diff) >= 8

    if flagged:
        note = ("Localized high recompression error: possible splice/clone/erase. "
                "Request the original uncropped file before any decision.")
    elif max_diff >= 60:
        note = "Globally high error (heavy prior compression). Likely benign; verify."
    else:
        note = "No localized manipulation signature above threshold."

    if save_map:
        # amplify for human viewing
        scale = 255.0 / max(max_diff, 1.0)
        vis = np.clip(arr * scale, 0, 255).astype(np.uint8)
        Image.fromarray(vis).save(save_map)

    return ELAResult(max_diff, mean_diff, hotspot_ratio, flagged, note)
