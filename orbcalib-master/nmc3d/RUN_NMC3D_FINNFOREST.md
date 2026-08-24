# Run NMC3D On FinnForest C1/C4 Atlases

This runs only the NMC3D calibration stage using saved FinnForest C1/C4
ORB-SLAM atlases. It does not replay the rosbag.

## Inputs

For the controlled workflow, expected atlas files are inside a run-specific
folder:

```text
results_finnforest/<run_id>/c1_atlasCamera 1.osa
results_finnforest/<run_id>/c4_atlasCamera 2.osa
```

`tools/run_finnforest_controlled_calib.sh` (the `../calib` calibration
helper) already produces atlases at exactly that path -- there is no export
or copy step, since `nmc3d/` reads the same `results_finnforest/<run_id>`
folder directly:

```bash
docker exec -it orbcalib bash -lc 'cd /ws/src/orbcalib-master && tools/run_finnforest_controlled_calib.sh --run-id 40Hz_Controlled_Slam'
```

## Configs

Use these FinnForest-specific configs (shared with `../calib`):

```text
config/sim/calib_finnforest.yaml
config/sim/C1.yaml
config/sim/C4.yaml
```

## Controlled Run-Id Workflow

From the host, use the same `--run-id` that you used for the controlled
`../calib` run:

```bash
docker exec -it orbcalib bash -lc '
  cd /ws/src/orbcalib-master
  nmc3d/run_finnforest_nmc3d_calib.sh --run-id 13Hz_Controlled
'
```

The helper:

1. verifies that the run folder contains the C1/C4 atlases,
2. creates temporary load configs in `results_finnforest/<run_id>/config/`,
3. points those configs at the run-specific atlas files,
4. builds the `calib_nmc3d` target,
5. starts a temporary `roscore`,
6. runs NMC3D calibration,
7. saves logs in the same run folder.

Expected outputs:

```text
results_finnforest/<run_id>/nmc3d_calib.log
results_finnforest/<run_id>/roscore_nmc3d_calib.log
results_finnforest/<run_id>/config/C1_nmc3d_calib_load.yaml
results_finnforest/<run_id>/config/C4_nmc3d_calib_load.yaml
```

To skip rebuilding once `calib_nmc3d` is already built:

```bash
docker exec -it orbcalib bash -lc '
  cd /ws/src/orbcalib-master
  nmc3d/run_finnforest_nmc3d_calib.sh --run-id 13Hz_Controlled --skip-build
'
```

## Ground-Plane Scale

The standalone `estimate_ground_scale.py` atlas-scale tool referenced by
earlier versions of this doc was retired; ground-plane scale recovery now
lives in `../tools/estimate_aruco_ground_plane_scale.py` (see
`../tools/README_scale_recovery.md`). It's part of the ArUco marker
trajectory-alignment pipeline and is not currently wired into the
CamMap/NMC3D map-alignment results here -- use it standalone if you need a
ground-plane-derived scale for a monocular atlas.

## One-Shot Docker Run (Manual Atlas Placement)

If you're not using the controlled-run-id workflow, place the atlases
directly in the repo root as `c1_atlasCamera 1.osa` / `c4_atlasCamera 2.osa`
and run:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --network host \
  --ipc=host \
  -e HOME=/tmp \
  -v "$HOME/Desktop/Dorsa/orbcalib-master":/ws/src/orbcalib-master \
  orbcalib-noetic bash -lc '
    source /opt/ros/noetic/setup.bash
    cd /ws/src/orbcalib-master
    cmake -S . -B build
    cmake --build build --target calib_nmc3d -j$(nproc)
    roscore >/tmp/roscore.log 2>&1 &
    sleep 3
    mkdir -p results_finnforest
    ./build/nmc3d/calib_nmc3d \
      ./Vocabulary/ORBvoc.txt \
      config/sim/calib_finnforest.yaml \
      config/sim/C1.yaml \
      config/sim/C4.yaml \
      2>&1 | tee results_finnforest/nmc3d_finnforest_c1_c4.log
  '
```

After the run, the result log should be writable from the host:

```bash
ls -lh "$HOME/Desktop/Dorsa/orbcalib-master/results_finnforest"
```

## What To Look For

The run should load:

```text
./c1_atlasCamera 1.osa
./c4_atlasCamera 2.osa
```

NMC3D-specific output includes lines such as:

```text
frame-to-frame distance-bin selection
global final depth selection
matched mps selected during frame-to-frame matching size
```

The final result, if calibration succeeds, is printed as:

```text
---- first pose optim ----
euler: ...
trans: ...
---- final pose optim ----
euler: ...
trans: ...
```

If it prints `no common features detected!!`, then NMC3D also failed to
accept geometrically valid C1/C4 common map regions from these atlases.
