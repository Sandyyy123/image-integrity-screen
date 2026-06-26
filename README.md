# image-integrity-screen

A small, dependency-light harness for **scientific-figure integrity screening** —
the first-pass triage an editorial integrity analyst runs on every submission
before a human review. It screens figure panels for **duplication** and
**manipulation**, then emits a severity-graded, JSON-able report with an
editorial recommendation.

> Screening aid, not a verdict. Every flag is meant to be **confirmed by a
> trained human reviewer** and, where warranted, by requesting the original
> uncropped source files from the authors. The goal is to surface candidates
> for review, not to accuse.

## What it checks

| Screen | Function | Catches |
|--------|----------|---------|
| Copy-move (within a panel) | `integrity.copy_move` | a region duplicated inside one figure (e.g. a blot pasted twice) |
| Cross-panel duplication | `integrity.cross_duplication` | a region reused across two panels/figures under different labels |
| Error-Level Analysis | `integrity.run_ela` | local splice / clone / erase via recompression mismatch |

Duplication uses **ORB keypoints + ratio-tested matching + a RANSAC homography
filter**, so only *geometrically consistent* match clusters survive — this is
what separates a genuine pasted region from incidental texture similarity, and
it is robust to flips, small rotations, and rescaling. ELA amplifies the
per-pixel JPEG-recompression error so locally edited blocks stand out.

## Quick start

```bash
pip install -r requirements.txt

# No image handy? Build a synthetic clean/tampered/shared set and run everything:
python main.py demo

# Screen one panel for internal duplication + manipulation:
python main.py screen path/to/figure.png

# Compare two panels for a reused region:
python main.py compare panelA.png panelB.png
```

### Demo output (abridged)

```
# clean panel               -> grade CLEAR
# patch pasted twice in one  -> copy_move_within_panel: 101 matches, FLAGGED -> QUERY AUTHORS
# same patch across 2 panels -> duplication_across_panels: 149 matches, FLAGGED -> QUERY AUTHORS
```

The synthetic demo is self-checking: the clean panel must come back CLEAR (true
negative) while both duplication cases must flag (true positives).

## Pipeline

```
ingest figure ─► split to panels ─► [copy-move] [cross-dup] [ELA] ─► severity grade ─► report + recommendation
                                          │            │        │
                                          └──── geometric / statistical evidence per flag ────┘
```

Recommendation tiers: `CLEAR` (proceed) · `QUERY AUTHORS` (hold, request
originals) · `HIGH CONCERN` (escalate: correction / expression of concern).

## Layout

```
main.py                 CLI: screen / compare / demo + synthetic generator
integrity/ela.py        Error-Level Analysis
integrity/duplication.py  ORB copy-move + cross-panel duplication
requirements.txt
```

## Scope and honesty

- Tuned to favour **recall** at the screening stage (better a human dismisses a
  false flag than a real duplication slips through). Thresholds are explicit and
  adjustable in each module.
- ELA is most informative on JPEG-derived figures; on lossless PNGs it reports
  honestly that no localized recompression signature is present.
- No conclusion of misconduct is ever made automatically. The tool documents
  *evidence*; the editorial decision stays with people.

Built by Dr. Sandeep Grover — PhD, 12 years biomedical research, peer-reviewed
author and manuscript reviewer.
