# ORB-SLAM3 Calibration Workspace

This folder contains the main code workspace for the project. For the high-level
project summary, related repositories, and paper citations, see the root
`../README.md`.

## Folder guide

- `calib/` -- baseline CamMap-style SLAM map-alignment calibration.
- `nmc3d/` -- depth-balanced match-selection variant.
- `orbslam3/` -- ORB-SLAM3 source used by the calibration pipelines.
- `tools/` -- scripts for controlled replay, atlas export, scale recovery,
  ArUco trajectory alignment, and reporting.
- `config/` -- example camera and calibration YAML files.
- `docker/` -- ROS Noetic build/runtime environment.

Generated atlases, recordings, logs, CSV outputs, PDFs, and result folders are
not versioned.

## Where to start

- SLAM map-alignment workflow: `RUN_DOCKER.md`
- Controlled image-folder workflow: `RUN_AGILEX_CONTROLLED.md`
- NMC3D variant: `nmc3d/README.md`
- Scale recovery: `tools/README_scale_recovery.md`
- ArUco trajectory calibration: `tools/README_trajectory_extrinsic_calibration.md`
- Clean ArUco workflow wrapper: `tools/README_clean_aruco_trajectory_calibration.md`

## Build

The tested build path is the ROS Noetic Docker environment in
`docker/noetic.Dockerfile`.

```bash
docker build -t orbcalib-noetic -f docker/noetic.Dockerfile .
cmake -B build
cmake --build build -- -j8
```

The ORB vocabulary is not committed because it is large. Download it from the
ORB-SLAM3 vocabulary release/source and place it at:

```text
Vocabulary/ORBvoc.txt
```

## Main local changes

- Added configurable per-camera global scale support in `calib/`.
- Hardened `Sim3Solver::ComputeSim3` against non-finite or degenerate RANSAC
  samples.
- Added atlas export/refinement utilities for scale-recovery and trajectory
  workflows.
- Integrated the `nmc3d/` depth-balanced calibration variant into the same build.
- Added controlled Agilex/FinnForest replay and calibration scripts.
