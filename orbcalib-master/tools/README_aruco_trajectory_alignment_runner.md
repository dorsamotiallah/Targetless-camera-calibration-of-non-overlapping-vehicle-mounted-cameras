# ArUco Trajectory Alignment Runner

`run_aruco_trajectory_alignment.sh` runs the full ArUco-based trajectory
alignment workflow for one two-camera ORB-SLAM run folder.

It runs:

1. `estimate_aruco_slam_scale.py` for camera 1
2. `estimate_aruco_slam_scale.py` for camera 2
3. `align_trajectories_to_aruco.py` using the generated ArUco point CSVs and
   extracted recommended scales

By default, trajectory matching uses original source PNG timestamps recovered
from `frame_pairs.csv`, not the ROS playback timestamps in the raw CSV. This
allows independently replayed runs to be compared on the same dataset clock.

## Required Inputs

The run folder must already contain:

```text
<run-dir>/manifest.txt
<run-dir>/<camera1>_raw_keyframe_observations.csv
<run-dir>/<camera2>_raw_keyframe_observations.csv
```

The raw CSVs must include pose columns:

```text
qw,qx,qy,qz
```

## Example

Run inside the `orbcalib` Docker container:

```bash
docker exec -it orbcalib bash -lc 'cd /ws/src/orbcalib-master && tools/run_aruco_trajectory_alignment.sh \
  --run-dir /ws/src/orbcalib-master/results_agilex_outdoor_latest/fablabDoor_front_left_defished_fov125_diag \
  --camera1 front \
  --camera2 left \
  --marker-id 0 \
  --marker-length-m 0.182 \
  --no-debug-images'
```

## Common Options

```text
--run-dir PATH                  Full run folder path
--run-id NAME                   Run folder under results_agilex_outdoor_latest
--camera1 NAME                  First camera name
--camera2 NAME                  Second camera name
--marker-id 0                   Use only ArUco marker ID 0
--marker-length-m 0.182         Marker side length in meters
--no-debug-images               Skip debug overlay images
--raw-ros-timestamps            Use legacy raw ROS playback timestamps
--max-distance-from-anchor-m M  Optional final alignment filter
--show                          Show the saved 3D plot interactively
```

## Outputs

ArUco scale outputs:

```text
<run-dir>/aruco_scale/<camera>/<camera>_aruco_scale_summary.json
<run-dir>/aruco_scale/<camera>/<camera>_aruco_scale_points.csv
<run-dir>/aruco_scale/<camera>/<camera>_aruco_scale_pairs.csv
```

Trajectory alignment outputs:

```text
<run-dir>/aruco_alignment/<camera1>_<camera2>_runner.log
<run-dir>/aruco_alignment/<camera1>_<camera2>_trajectories.png
<run-dir>/aruco_alignment/<camera1>_<camera2>_alignment.json
<run-dir>/aruco_alignment/<camera1>_<camera2>_extrinsic.yaml
```
