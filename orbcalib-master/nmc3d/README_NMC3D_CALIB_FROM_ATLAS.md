# NMC3D Calibration From Saved Atlases

`nmc3d/` builds as part of this same project (see `../README.md` and
`README.md` in this folder) -- there is no separate repo, image, or
container for it. Use it to test the NMC3D-style depth-balanced feature
selection without changing the original CamMap code in `../calib`.

The workflow below assumes SLAM has already been run and the atlas files
already exist. It only runs the calibration stage, in the same `orbcalib`
container you already use for `../calib`.

## 1. Build the `calib_nmc3d` Target

Run this after every source-code change before launching calibration.

```bash
docker exec -it orbcalib bash -lc '
  cd /ws/src/orbcalib-master &&
  source /opt/ros/noetic/setup.bash &&
  cmake -S . -B build &&
  cmake --build build --target calib_nmc3d -j$(nproc)
'
```

This is the exact same `build/` directory used for `../calib`'s `calib`
target and the ArUco tools' `atlas_export_*` targets -- one build tree for
the whole project.

## 2. Atlas Files

The camera YAML files currently load:

```yaml
System.LoadAtlasFromFile: "front_atlas"
System.LoadAtlasFromFile: "back_atlas"
```

ORB-SLAM expects these files next to wherever you invoke the binary from,
with the camera suffix:

```text
front_atlasCamera 1.osa
back_atlasCamera 2.osa
```

If you already produced these atlases with `../calib`'s `calib` binary or
with `tools/run_agilex_controlled_slam.sh` / `tools/run_finnforest_controlled_slam.sh`,
they're already sitting in that run's result folder -- there's nothing to
copy. Just `cd` there before running `calib_nmc3d`, or point `RUN_DIR` at it
(see `RUN_NMC3D_FINNFOREST.md` for the controlled-run-folder workflow).

## 3. Check Calibration Mode

Make sure `config/sim/calib.yaml` is in calibration mode:

```yaml
Mode: calib
```

Make sure the camera configs are loading, not saving, atlases:

```yaml
System.LoadAtlasFromFile: "front_atlas"
# System.SaveAtlasToFile: "front_atlas"
```

```yaml
System.LoadAtlasFromFile: "back_atlas"
# System.SaveAtlasToFile: "back_atlas"
```

## 4. Run Calibration

Even in `Mode: calib`, the executable calls `ros::start()`, so a ROS master
must be running. Keep one container running, start `roscore` in one
terminal, then run the calibration executable in another with `docker exec`
(same pattern as `../calib`'s `calib`):

Terminal 1, inside the container:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

Terminal 2, rebuild and run after every code change:

```bash
docker exec -it orbcalib bash -lc '
  cd /ws/src/orbcalib-master &&
  source /opt/ros/noetic/setup.bash &&
  cmake -S . -B build &&
  cmake --build build --target calib_nmc3d -j$(nproc) &&
  export LIBGL_ALWAYS_SOFTWARE=1 &&
  ./build/nmc3d/calib_nmc3d \
    ./Vocabulary/ORBvoc.txt \
    config/sim/calib.yaml \
    config/sim/front_cam.yaml \
    config/sim/back_cam.yaml
'
```

For a one-shot run (build, start `roscore`, run, all in one `docker run`),
mirror the pattern in `../RUN_DOCKER.md`, swapping `./build/calib/calib` for
`./build/nmc3d/calib_nmc3d` and `--target calib` for `--target calib_nmc3d`.

## 5. What To Look For

The NMC3D code prints messages like:

```text
global final depth selection: valid ..., near ..., middle ..., far ..., selected ...
global 8x8 grid cells occupied: near ..., middle ..., far ...; selected near ..., middle ..., far ...
matched mps from CamMap matching size: ...
matched mps selected for optim size: ...
```

This means the current workflow was compiled and is running: CamMap keyframe
matching is unchanged, then one global final depth/grid selection is applied
before optimization.

If it prints:

```text
local adaptive depth selection skipped: near ..., middle ..., far ...
```

then one depth group did not have enough matches, so the code kept the
original CamMap matches for that keyframe pair.

If it still prints the older messages:

```text
local quantile depth selection: valid ..., shallow ..., middle ..., deep ...
local adaptive depth selection: valid ..., near ..., middle ..., far ...
matched mps selected for global optim size: ...
matched mps all-depth for optim size: ...
```

then you are running a stale binary -- rebuild with the command in Section 4.

The final calibration output is still:

```text
inliers size: ...
---- first pose optim ----
euler: ...
trans: ...
---- final pose optim ----
euler: ...
trans: ...
```

Use these values to compare against `../calib`'s `calib` result on the same
atlases.
