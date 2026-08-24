## Non-Overlap Camera Pair Calibration: Depth-Balanced Match Selection

### Intro.

Extends the CamMap-based calibration in `../calib` with the
uniformly-distributed ("depth-balanced") match-selection idea from NMC3D [1]:
after CamMap's own matching, matched map-point pairs are grouped into
near/middle/far distance bins and spread across an image grid before being
used in the final Sim3/reprojection optimization, instead of being used
as-is. This keeps the optimizer from being dominated by whichever depth
range happened to produce the most raw matches.

This folder builds as part of the top-level project (see `../README.md` for
dependencies, build, and the shared ORB-SLAM3 modifications) -- it is not a
separate checkout. `CMakeLists.txt` here defines two additional executables
alongside the ones in `../calib`, sharing the same `ORB_SLAM3` library:

- `calib_nmc3d` (from `main.cpp` + `calib.cpp`, plus `../calib/edge.cpp`
  referenced in place rather than duplicated) -- the depth-balanced
  calibration node. Usage matches `../calib`'s `calib` node (same
  `calib.yaml`/`cam.yaml` config, same `slam`/`calib` mode split); just
  invoke `./build/nmc3d/calib_nmc3d` instead of `./build/calib/calib`.
- `atlas_ground_export` -- ground-plane-relevant atlas export used while
  developing this variant.

### Running It

See `README_NMC3D_CALIB_FROM_ATLAS.md` for the manual/atlas-in-place
workflow, or `RUN_NMC3D_FINNFOREST.md` for the controlled-run-id workflow on
FinnForest C1/C4 recordings (shared `results_finnforest/<run_id>` folders
with `../calib`, no separate export step).

### Performance

See `calibration_approaches_summary.md` for measured results against the
CamMap baseline (translation/rotation error vs. ground truth, inlier counts)
across the tested recordings, and for the archived method variants in
`calibration_versions/` that were tried en route to the current `calib.cpp`.
Saved outputs from those earlier (pre-merge) experiments -- not regenerated
by the current scripts, kept for reference -- live under `results/`.

### Ref.

[1] C. Dai, T. Han, Y. Luo, M. Wang, G. Cai, J. Su, Z. Gong, and N. Liu,
"NMC3D: Non-Overlapping Multi-Camera Calibration Based on Sparse 3D Map,"
Sensors 2024, 24, 5228, doi: 10.3390/s24165228.
