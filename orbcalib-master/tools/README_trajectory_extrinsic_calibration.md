# Trajectory-Based Extrinsic Calibration via ArUco

`align_trajectories_to_aruco.py` calibrates two cameras whose **independent
monocular SLAM sessions never share any physical scene content** to match
features between (e.g. front vs. left, front vs. back, when neither camera's
path ever revisits anywhere the other one mapped) -- using a shared ArUco
marker as the common reference frame instead of feature matching between the
two maps. If your maps do share some scene content, `/calib`'s CamMap-based
map alignment (`../README.md`) needs no marker at all and is the simpler
default; use this tool when that isn't an option.

Run it after `estimate_aruco_slam_scale.py` (see `README_scale_recovery.md`)
and after re-exporting the raw keyframe CSVs with the pose-carrying
`atlas_export_observations` build (adds `qw,qx,qy,qz` columns).

When a raw CSV is inside a controlled-run folder with `manifest.txt` and
`frame_pairs.csv`, the tool converts raw ROS playback timestamps back to the
original source PNG timestamps before matching trajectories. This makes
independently replayed/single-camera runs comparable on the dataset clock.
Use `--raw-ros-timestamps` only for legacy/debug runs where the raw CSV
timestamps should be used directly.

## `--alignment-source` (required)

Stage 1 (below) can register each camera's SLAM map to the marker frame in
one of three ways. There is no default -- pick one explicitly:

- **`optimized-anchor`** (recommended whenever you already have a per-camera
  scale from elsewhere, e.g. `estimate_aruco_slam_scale.py`'s point-pair
  scale or `estimate_aruco_ground_plane_scale.py`'s ground-plane scale).
  Detects the marker in every exported keyframe it can, solves metric PnP
  for each detection, and fits rotation as a MAD-trimmed, iteratively
  re-weighted chordal mean across *all* of them -- while holding scale fixed
  at the externally supplied `--camera1-scale`/`--camera2-scale`. Requires
  both scales; there is no per-camera fallback.
- **`visual-aruco-sim3`** -- same many-keyframe marker detection and
  rotation fit as `optimized-anchor`, but also fits its own scale jointly
  (least-squares over camera-center spread in SLAM units vs. marker-frame
  units). Use this when you don't have an external scale for one or both
  cameras. Its scale fit is sensitive to monocular SLAM scale drift over
  long, spatially separated marker sightings (see `README_scale_recovery.md`
  discussion of point-pair vs. Sim(3) scale) -- prefer `optimized-anchor`
  with a point-pair or ground-plane scale when you have one.
- **`visual-aruco-motion-scale`** -- same many-keyframe detections, but
  recovers scale from robust pairwise camera-motion-distance ratios instead
  of the Sim(3) least-squares fit. Experimental; not used in the main
  Ground/PointPair/Sim3 scale-method comparison.

## What It Does

One tool, three stages, run together (each reuses what the previous one
computes -- no need to invoke them separately):

1. **Alignment**, via one of the three `--alignment-source` modes above.
   Every mode ends with one similarity transform (scale, rotation,
   translation) per camera, applied to every keyframe to give that camera's
   full trajectory in the shared, metric, marker-centered frame. Plots both
   trajectories.

2. **Extrinsic calibration**, reusing stage 1's trajectories. Since the two
   cameras share no visual content, keyframes are matched by timestamp
   instead: for every camera2 keyframe, camera1's pose is interpolated
   (SLERP + linear) to that exact timestamp, and the relative pose is
   computed at every match. Matches are optionally filtered by distance/time
   from each camera's reference keyframe (see Notes), and the survivors are
   robustly averaged (SVD chordal mean for rotation, per-axis median for
   translation) into the "averaged" extrinsic.

3. **Joint optimization refinement**, reusing stage 2's surviving matches, in
   two variants:

   - **3a (per-component, `scipy.optimize.least_squares`).** Fits a single
     (R, t) minimizing one combined robust (Huber by default) loss over
     rotation and translation residuals *together*, across all matches at
     once -- seeded from stage 2's result. The robust loss is applied to
     each of the 6 residual numbers per match (3 rotation + 3 translation)
     independently, because that's what `least_squares`'s API supports.
   - **3b (grouped, `scipy.optimize.minimize`, L-BFGS-B).** Same idea, but
     the robust loss is applied once to each match's *combined* 6D residual
     norm, so a match with a large combined error is downweighted or
     rejected as a whole rather than having each of its 6 numbers judged
     independently. This is the thing a generic scalar minimizer (the
     Python analog of MATLAB's `fminunc`) can do that `least_squares`'s
     per-component loss structurally cannot.

   Outlier matches are smoothly downweighted instead of only relying on the
   hard distance/time cutoffs. All three results (averaged, optimized,
   optimized-grouped) are printed and saved side by side (as
   `T_<camera1>_<camera2>`, `T_<camera1>_<camera2>_optimized`, and
   `T_<camera1>_<camera2>_optimized_grouped`) so they can be compared run to
   run -- neither optimizer is assumed to be better a priori.

## Inputs Required

Per camera:

```text
results_agilex/<run_id>/aruco_scale/<camera>/<camera>_aruco_scale_points.csv   # from estimate_aruco_slam_scale.py
results_agilex/<run_id>/<camera>_raw_keyframe_observations.csv                 # must include qw,qx,qy,qz columns
config/sim/agilex_<camera>_defished_cam.yaml                                   # camera intrinsics
```

If a camera's raw CSV predates the pose export fix, regenerate it:

```bash
./build/calib/atlas_export_observations \
  Vocabulary/ORBvoc.txt \
  results_agilex/<run_id>/config/<camera>_controlled_observation_export_load.yaml \
  "Camera 1" \
  results_agilex/<run_id>/<camera>_raw_keyframe_observations.csv
```

(`"Camera 1"` for camera1, `"Camera 2"` for camera2 -- must match the atlas
prefix used when the run was recorded.)

## Usage

```bash
python3 tools/align_trajectories_to_aruco.py \
  --alignment-source optimized-anchor \
  --camera1-name front \
  --camera1-points-csv results_agilex/<run_id>/aruco_scale/front/front_aruco_scale_points.csv \
  --camera1-raw-csv results_agilex/<run_id>/front_raw_keyframe_observations.csv \
  --camera1-config config/sim/agilex_front_defished_cam.yaml \
  --camera1-scale 5.864969842432022 \
  --camera2-name left \
  --camera2-points-csv results_agilex/<run_id>/aruco_scale/left/left_aruco_scale_points.csv \
  --camera2-raw-csv results_agilex/<run_id>/left_raw_keyframe_observations.csv \
  --camera2-config config/sim/agilex_left_defished_cam.yaml \
  --camera2-scale 4.0349409367671125 \
  --max-distance-from-anchor-m 1.0
```

Run once *without* `--max-distance-from-anchor-m` first: the tool prints a
sensitivity table (matches kept / rotation deviation / translation deviation
at several cutoffs) so you can pick a cutoff deliberately instead of
guessing. Then rerun with your chosen cutoff for the final saved extrinsic.

Useful options:

```text
--marker-id 3                     # force a marker instead of auto-selecting the best-shared one
--marker-length-m 0.182            # marker side length, for the plot's marker outline only
--camera1-scale 5.864969842432022  # camera1's SLAM map scale (m per SLAM unit), from
                                    # estimate_aruco_slam_scale.py, estimate_aruco_ground_plane_scale.py,
                                    # or a MATLAB ground-plane fit. Required for --alignment-source
                                    # optimized-anchor; ignored (and unnecessary) for visual-aruco-sim3,
                                    # which fits its own scale.
--camera2-scale 4.0349409367671125 # same as --camera1-scale, for camera2
--visual-aruco-min-detections 4    # minimum many-keyframe marker detections needed (sim3/optimized-anchor)
--visual-aruco-max-rms-px 3.0      # reject a keyframe's marker PnP above this reprojection RMS
--visual-aruco-trim-mad 3.5        # MAD cutoff for trimming rotation-fit outliers, 0 disables trimming
--hide-anchor-marker               # don't draw the single reference-keyframe star in the trajectory plot
--max-bracket-width-s 0.5          # drop matches where camera1's bracketing keyframes are farther
                                    # apart in time than this (usually matters little, see Notes)
--max-distance-from-anchor-m 1.0   # drop matches farther than this from either camera's reference
                                    # keyframe (the main quality knob, see Notes)
--show                             # also display the plot interactively (it is always saved)
--output-plot PATH                 # override the default save location (see below)
--output-alignment-json PATH       # override the default save location (see below)
--output-extrinsic-yaml PATH       # override the default save location (see below)
--optimize-loss huber              # robust loss for stage 3: linear/huber/soft_l1/cauchy (default: huber)
--optimize-f-scale-m 0.02          # stage 3's robust-loss transition scale (meters-equivalent). Default:
                                    # 1.5x the median translation deviation from the stage-2 average.
```

## Outputs

Always saved (no flags required), all under `<run-dir>/aruco_alignment/`,
where `<run-dir>` is `--camera1-raw-csv`'s parent directory:

```text
results_agilex/<run_id>/aruco_alignment/<camera1>_<camera2>_trajectories.png   # stage 1: 3D plot of both trajectories
results_agilex/<run_id>/aruco_alignment/<camera1>_<camera2>_alignment.json     # stage 1: marker/keyframes/scale/Sim3 per camera
results_agilex/<run_id>/aruco_alignment/<camera1>_<camera2>_extrinsic.yaml     # stage 2 + 3: both T_ blocks, plus diagnostics
```

The extrinsic YAML contains **two** blocks, both in the format used in
`Agilex Recordings/Intrinsic_ground_truths/robot_relative_extrinsics.yaml`
exactly -- `from_frame`/`to_frame`, `translation_xyz`, `euler_zyx_deg`
(`roll`/`pitch`/`yaw`, ZYX convention: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`),
and the full 4x4 homogeneous `matrix`:

- `T_<camera1>_<camera2>` -- stage 2's averaged result.
- `T_<camera1>_<camera2>_optimized` -- stage 3a's per-component joint-optimization result.
- `T_<camera1>_<camera2>_optimized_grouped` -- stage 3b's grouped joint-optimization result.

Any of the three can be dropped straight into the ground-truth file or
diffed against it. A `calibration_diagnostics` section (match counts,
cutoffs used, rotation/translation deviation stats, and `optimization` /
`optimization_grouped` sub-sections -- loss/robust scale used, cost
before/after, how far each stage-3 variant moved from stage 2) is appended
below all three blocks, outside them.

Also printed to stdout: the marker-selection table, per-camera PnP/scale
diagnostics, the bracket-width and distance-from-anchor sensitivity tables,
and all three final `T_<camera1>_<camera2>*` blocks.

## Notes

- **Marker coverage matters.** Pick (or let auto-selection pick) a marker
  actually seen by both cameras -- check the printed per-marker keyframe
  counts. A marker only one camera saw is useless for this.
- **`optimized-anchor` and `visual-aruco-sim3` expose one "reference"
  keyframe** (the lowest-reprojection-RMS detection, `anchor_point`/`kf_id`
  in the alignment JSON) for plotting and logging, but it does not define
  the transform by itself -- rotation is the robust average over every kept
  keyframe. Distance/time cutoffs are measured from that reference point as
  a practical proxy for "near the marker."
- **Distance from the reference keyframe, not timestamp gap, is usually the
  dominant accuracy driver**, because monocular SLAM drift grows with
  distance traveled and has no other metric constraint over a long loop.
  Always check the `--max-distance-from-anchor-m` sensitivity table rather
  than assuming a default cutoff is right for a new dataset.
- **Validate against any ground truth you have** when one is available --
  it's the only reliable check against a systematic bias (as opposed to
  noise) in the pipeline, which an internal consistency check (spread across
  matches) alone will not reveal.
- **Stage 3 usually lands close to stage 2** on clean data (no outlier
  matches) -- that's expected, since Huber loss behaves like ordinary least
  squares near the bulk of the data and only downweights far-out residuals.
  If stage 3 moves far from stage 2, treat that as a signal to revisit the
  `--max-distance-from-anchor-m` cutoff rather than assuming the optimizer
  found a better answer.
- **Scale is never guessed for `optimized-anchor`.** It always comes from
  `--camera1-scale`/`--camera2-scale` -- see the point-pair method in
  `estimate_aruco_slam_scale.py`, the ground-plane method in
  `estimate_aruco_ground_plane_scale.py`, or `README_scale_recovery.md`.
  `visual-aruco-sim3` is the only mode that recovers scale on its own (per
  camera, from the same many-keyframe fit as its rotation); pass
  `--camera1-scale`/`--camera2-scale` to `visual-aruco-sim3` and they are
  simply unused. The alignment JSON records which source was used per
  camera (`scale_source`: `"external_optimized_anchor"` or the Sim(3)/motion
  fit's own recovered value).
