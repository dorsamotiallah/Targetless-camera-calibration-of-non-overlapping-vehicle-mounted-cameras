# Clean ArUco Trajectory Calibration Runner

Use the separated runner scripts when ORB-SLAM has already been run and you
want scale recovery and calibration to be explicit checkpoints.

Scale output structure:

```text
<scale-output-root>/
  aruco_points_scale/<camera>/
  ground_scale/
  sim3_scale/
  scale_report.json
  scale_report.log
```

Calibration output structure:

```text
<run-dir>/
  calibration_aruco_points_scale/aruco_alignment/
  calibration_ground_scale/aruco_alignment/
  calibration_sim3_scale/aruco_alignment/
```

Default marker policy:

- Scale extraction uses all detected markers.
- Trajectory alignment needs one marker coordinate frame. If `--marker-id` is
  omitted, the alignment tool picks the best marker shared by both cameras.
- Add `--marker-id ID` only when you intentionally want to force one marker.

Step 1: recover and report scales only:

```bash
python3 tools/run_clean_aruco_scale_recovery.py \
  --run-dir results_fisheye/LobbyZigZag4Tags/front_back \
  --camera1 front \
  --camera2 back \
  --camera1-config config/sim/agilex_front_cam.yaml \
  --camera2-config config/sim/agilex_back_cam.yaml \
  --output-root results_fisheye/LobbyZigZag4Tags/front_back/scales \
  --scale-mode all
```

This writes `scale_report.log` and `scale_report.json`, and saves each scale
source separately.

Step 2: run calibration using one chosen scale source:

```bash
python3 tools/run_clean_aruco_calibration_from_scale.py \
  --run-dir results_fisheye/LobbyZigZag4Tags/front_back \
  --camera1 front \
  --camera2 back \
  --camera1-config config/sim/agilex_front_cam.yaml \
  --camera2-config config/sim/agilex_back_cam.yaml \
  --scale-root results_fisheye/LobbyZigZag4Tags/front_back/scales \
  --scale-source ground
```

Use `--scale-source aruco-points`, `--scale-source ground`, or
`--scale-source sim3`.

The old one-command wrappers were removed. Use this two-step workflow so scale
recovery and calibration results stay separate and inspectable.
