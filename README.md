# Non-Overlapping Vehicle Camera Calibration

This repository contains my work on extrinsic calibration for vehicle-mounted
cameras with little or no shared field of view. The code is organized around two
main calibration pipelines: SLAM-map alignment and ArUco-assisted trajectory
alignment.

## What is in this repo?

- `orbcalib-master/` -- main C++/Python workspace built around ORB-SLAM3 and
  the [CamMap](#key-papers)-style non-overlapping camera calibration pipeline.
- `orbcalib-master/calib/` -- baseline SLAM map-alignment calibration adapted
  from `Rick0514/orbcalib`.
- `orbcalib-master/nmc3d/` -- depth-balanced match-selection experiments
  inspired by [NMC3D](#key-papers), including implementation variants and
  method notes.
- `orbcalib-master/tools/` -- scripts for controlled runs, ArUco-based scale
  recovery, trajectory alignment, reporting, and evaluation utilities.
- `orbcalib-master/config/` -- camera and calibration YAML examples for the
  tested simulated, Agilex, and FinnForest setups.
- Generated outputs, saved atlases, recordings, and local paper PDFs are kept
  out of version control; the repository is intended to contain code,
  configuration, and documentation only.

## Implemented methods

### SLAM map alignment

Each camera builds an independent ORB-SLAM3 map. The maps are then matched and
aligned to estimate the relative camera transform. Start here:

- Build and general usage: `orbcalib-master/README.md`
- Docker/ROS workflow: `orbcalib-master/RUN_DOCKER.md`
- Controlled image-folder workflow: `orbcalib-master/RUN_AGILEX_CONTROLLED.md`
- NMC3D-inspired variant: `orbcalib-master/nmc3d/README.md`

### ArUco trajectory alignment

After testing the SLAM map-alignment approach across a wide range of cases, we
found that it could not reliably find enough correct matches between the
reconstructed 3D map points or align the maps accurately enough. The ArUco
workflow was developed as an alternative: each trajectory is registered to a
common ArUco marker frame and matched by timestamp. Scale can be recovered from
marker geometry, [ground-plane height](#key-papers), or a robust Sim(3) fit.
Start here:

- Scale recovery: `orbcalib-master/tools/README_scale_recovery.md`
- Trajectory extrinsic calibration: `orbcalib-master/tools/README_trajectory_extrinsic_calibration.md`
- Clean end-to-end workflow: `orbcalib-master/tools/README_clean_aruco_trajectory_calibration.md`

## Related repositories

- [jiejie567/SlamForCalib](https://github.com/jiejie567/SlamForCalib) --
  official CamMap implementation released by the CamMap authors.
- [Rick0514/orbcalib](https://github.com/Rick0514/orbcalib) -- cleaner
  CamMap-based repository referenced by the official implementation; this
  project was originally forked from it.
- [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) -- SLAM
  system used as the base mapping and tracking framework.

## Key papers

- J. Xu et al., “[CamMap: Extrinsic Calibration of Non-Overlapping Cameras Based
  on SLAM Map Alignment](https://doi.org/10.1109/LRA.2022.3207793),” IEEE
  Robotics and Automation Letters, 2022.
- C. Dai et al., “[NMC3D: Non-Overlapping Multi-Camera Calibration Based on
  Sparse 3D Map](https://doi.org/10.3390/s24165228),” Sensors, 2024.
- C.-H. Ma, C.-M. Hsu, and J.-H. Chou, “[Scale Estimation for Monocular Visual
  Odometry Using Reliable Camera Height](https://doi.org/10.1109/SMC53654.2022.9945178),”
  IEEE SMC, 2022.
