#!/usr/bin/env python3
"""image-integrity-screen — a screening harness for scientific-figure integrity.

Runs three forensic screens on figure images and prints a severity-graded,
JSON-able report with an editorial recommendation:

  * copy-move duplication WITHIN a panel        (integrity.copy_move)
  * duplication ACROSS two panels/figures        (integrity.cross_duplication)
  * error-level analysis for splice/clone/erase  (integrity.run_ela)

USAGE
  # screen a single panel for internal duplication + manipulation
  python main.py screen path/to/figure.png

  # compare two panels for a reused region
  python main.py compare panelA.png panelB.png

  # no image handy? build a synthetic test pair and run the whole pipeline
  python main.py demo

The verdicts are screening aids. Every flag is meant to be confirmed by a
trained human reviewer and, where relevant, by requesting the original
uncropped source files from the authors.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from integrity import run_ela, copy_move, cross_duplication


def _grade(flags: int) -> str:
    return {0: "CLEAR", 1: "QUERY AUTHORS"}.get(flags, "HIGH CONCERN")


def _recommend(grade: str) -> str:
    return {
        "CLEAR": "Proceed; log screening parameters to the audit trail.",
        "QUERY AUTHORS": "Hold decision; request originals / clarification before acceptance.",
        "HIGH CONCERN": "Escalate: multiple flags. Consider correction / expression of concern.",
    }[grade]


def screen_one(path: str) -> dict:
    cm = copy_move(path)
    ela = run_ela(path)
    flags = int(cm.flagged) + int(ela.flagged)
    grade = _grade(flags)
    return {
        "manuscript_target": path,
        "checks": [cm.as_dict(), ela.as_dict()],
        "flags": flags,
        "grade": grade,
        "recommendation": _recommend(grade),
    }


def compare_two(a: str, b: str) -> dict:
    cd = cross_duplication(a, b)
    ela_a, ela_b = run_ela(a), run_ela(b)
    flags = int(cd.flagged) + int(ela_a.flagged) + int(ela_b.flagged)
    grade = _grade(flags)
    return {
        "panels": [a, b],
        "checks": [cd.as_dict(), ela_a.as_dict(), ela_b.as_dict()],
        "flags": flags,
        "grade": grade,
        "recommendation": _recommend(grade),
    }


def _textured_panel(seed: int, size=(240, 320)) -> np.ndarray:
    """A unique noisy background with distinct shapes — enough keypoints for ORB."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 28, (size[0], size[1], 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    for _ in range(14):
        x, y = int(rng.integers(0, size[1] - 30)), int(rng.integers(0, size[0] - 30))
        c = tuple(int(v) for v in rng.integers(20, 230, 3))
        if rng.random() < 0.5:
            d.ellipse([x, y, x + 24, y + 18], outline=c, width=2)
        else:
            d.line([x, y, x + 26, y + 22], fill=c, width=2)
    return np.asarray(img).copy()


def _texture_stamp(seed=7, h=56, w=56) -> np.ndarray:
    """A distinctive textured patch we will duplicate (the 'reused band')."""
    rng = np.random.default_rng(seed)
    patch = rng.normal(110, 35, (h, w, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(patch)
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 46, 40], outline=(20, 20, 20), width=3)
    d.line([4, 50, 52, 6], fill=(230, 40, 40), width=3)
    d.rectangle([18, 22, 40, 44], outline=(40, 220, 60), width=2)
    return np.asarray(img).copy()


def _make_synthetic(tmpdir: str = ".") -> tuple[str, str]:
    """Build (a) a clean panel and (b) a tampered panel where a textured patch is
    duplicated WITHIN it; and a third, different panel that shares that same patch
    (written to disk) to demonstrate cross-panel reuse."""
    stamp = _texture_stamp()
    h, w = stamp.shape[:2]

    # clean panel A
    a = _textured_panel(seed=1)
    clean = f"{tmpdir}/panel_clean.png"
    Image.fromarray(a).save(clean)

    # tampered panel A: same patch pasted at TWO locations (within-panel duplication)
    a_t = a.copy()
    a_t[40:40 + h, 30:30 + w] = stamp
    a_t[150:150 + h, 230:230 + w] = stamp
    tampered = f"{tmpdir}/panel_tampered.png"
    Image.fromarray(a_t).save(tampered)

    # panel B: a DIFFERENT background that shares the same patch once (cross-panel reuse)
    b = _textured_panel(seed=99)
    b[90:90 + h, 120:120 + w] = stamp
    Image.fromarray(b).save(f"{tmpdir}/panel_B_shared.png")

    return clean, tampered


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    cmd = argv[1]
    if cmd == "screen" and len(argv) >= 3:
        report = screen_one(argv[2])
    elif cmd == "compare" and len(argv) >= 4:
        report = compare_two(argv[2], argv[3])
    elif cmd == "demo":
        clean, tampered = _make_synthetic()
        shared = "./panel_B_shared.png"
        print("# Synthetic clean panel (expect: CLEAR):")
        print(json.dumps(screen_one(clean), indent=2))
        print("\n# Tampered panel — same patch pasted twice within it (expect: copy-move flag):")
        print(json.dumps(screen_one(tampered), indent=2))
        print("\n# Cross-panel reuse — different backgrounds, one shared patch (expect: flag):")
        report = compare_two(tampered, shared)
    else:
        print(__doc__)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
