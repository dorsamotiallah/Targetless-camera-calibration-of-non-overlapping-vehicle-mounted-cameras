#!/usr/bin/env python3
"""Recover a visual-ArUco Sim(3) scale using every available marker, not just one.

fit_visual_aruco_sim3() (in align_trajectories_to_aruco.py, used unmodified here)
fits a full Sim(3) -- rotation, translation, and scale -- anchored to ONE marker's
own local frame. Different markers sit at different physical locations, so their
"center_marker" observations live in different, unrelated coordinate frames and
cannot be pooled directly the way point-pair or ground-plane observations can
(those only ever compare a marker against its own known size/height, so they
never need to relate one marker to another).

Scale, however, is a single physical property of one camera's monocular map
("meters per SLAM unit") and does not depend on which marker you used to recover
it. So this script runs the existing per-marker Sim(3) fit independently for
every marker with enough observations, then combines the resulting SCALE values
only (never the per-marker rotations/translations, which are not comparable
across markers) into one robust combined estimate.

This intentionally does not modify run_clean_aruco_scale_recovery.py or
align_trajectories_to_aruco.py -- it only imports their existing functions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from align_trajectories_to_aruco import (
    collect_visual_aruco_pose_observations,
    fit_visual_aruco_sim3,
    load_trajectory,
)
from estimate_checkerboard_camera_height import read_calibration
from clean_aruco_workflow import default_raw_csv, require_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--camera1-run-dir", type=Path, help="Overrides --run-dir for camera 1.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--camera-config", required=True, type=Path)
    parser.add_argument("--raw-csv", type=Path, help="Defaults to <run-dir>/<camera>_raw_keyframe_observations.csv")
    parser.add_argument(
        "--points-csv",
        type=Path,
        help="Existing <camera>_aruco_scale_points.csv (from estimate_aruco_slam_scale.py). "
        "Used only to discover which marker ids are present and how many keyframes each has. "
        "Not required if --marker-id is given explicitly.",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        action="append",
        help="Restrict to these marker ids. Can be repeated. If omitted, every marker id found "
        "in --points-csv with enough keyframes is attempted.",
    )
    parser.add_argument(
        "--min-keyframes-per-marker",
        type=int,
        default=4,
        help="Minimum keyframes (from --points-csv) before a marker id is even attempted. Default: 4",
    )
    parser.add_argument("--marker-length-m", type=float, default=0.182)
    parser.add_argument("--dictionary-size", type=int, default=50)
    parser.add_argument("--dictionary-bits", type=int, default=6)
    parser.add_argument("--visual-aruco-min-detections", type=int, default=4)
    parser.add_argument("--visual-aruco-max-rms-px", type=float, default=3.0)
    parser.add_argument("--visual-aruco-trim-mad", type=float, default=3.5)
    parser.add_argument("--max-keyframes", type=int, help="Cap on keyframes scanned per marker.")
    parser.add_argument("--raw-ros-timestamps", action="store_true")
    parser.add_argument("--out-dir", type=Path, help="Defaults to <run-dir>/sim3_scale_all_markers/<camera>")
    return parser.parse_args()


def marker_keyframe_counts(points_csv: Path) -> Dict[int, int]:
    counts: Dict[int, set] = defaultdict(set)
    with points_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            counts[int(row["marker_id"])].add(row["kf_id"])
    return {marker_id: len(kfs) for marker_id, kfs in counts.items()}


def main() -> int:
    args = parse_args()
    run_dir = require_run_dir(args, "camera1")
    raw_csv = args.raw_csv or default_raw_csv(run_dir, args.camera)
    out_dir = args.out_dir or (args.run_dir / "sim3_scale_all_markers" / args.camera)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = read_calibration(args.camera_config)
    use_source_timestamps = not args.raw_ros_timestamps
    trajectory, timestamp_source = load_trajectory(raw_csv, args.camera, use_source_timestamps)

    if args.marker_id:
        marker_ids = sorted(set(args.marker_id))
        marker_kf_counts = {mid: None for mid in marker_ids}
    else:
        if not args.points_csv:
            raise SystemExit("Provide --marker-id (repeatable) or --points-csv to discover marker ids.")
        counts = marker_keyframe_counts(args.points_csv)
        marker_kf_counts = {mid: n for mid, n in counts.items() if n >= args.min_keyframes_per_marker}
        marker_ids = sorted(marker_kf_counts, key=lambda mid: -marker_kf_counts[mid])
        if not marker_ids:
            raise SystemExit(f"No marker in {args.points_csv} has >= {args.min_keyframes_per_marker} keyframes.")

    print(f"{args.camera}: trajectory timestamps = {timestamp_source}")
    print(f"{args.camera}: attempting marker ids {marker_ids} (from {'--marker-id' if args.marker_id else args.points_csv})")

    per_marker: List[Dict[str, object]] = []
    for marker_id in marker_ids:
        entry: Dict[str, object] = {"marker_id": marker_id, "keyframes_with_marker": marker_kf_counts[marker_id]}
        try:
            observations = collect_visual_aruco_pose_observations(
                args.camera,
                raw_csv,
                marker_id,
                calib,
                args.marker_length_m,
                args.dictionary_size,
                args.dictionary_bits,
                trajectory,
                args.visual_aruco_max_rms_px,
                args.max_keyframes,
            )
            scale, R, t, keep, diagnostics = fit_visual_aruco_sim3(
                f"{args.camera}[marker {marker_id}]",
                observations,
                args.visual_aruco_min_detections,
                args.visual_aruco_trim_mad,
            )
            entry.update(
                {
                    "status": "ok",
                    "detections": len(observations),
                    "detections_kept": int(np.count_nonzero(keep)),
                    "scale_m_per_slam_unit": scale,
                    "diagnostics": diagnostics,
                }
            )
            print(
                f"{args.camera} marker {marker_id}: scale={scale:.6f} m/slam-unit "
                f"using {int(np.count_nonzero(keep))}/{len(observations)} detection(s)"
            )
        except ValueError as exc:
            entry.update({"status": "skipped", "reason": str(exc)})
            print(f"{args.camera} marker {marker_id}: skipped ({exc})")
        per_marker.append(entry)

    ok_entries = [e for e in per_marker if e["status"] == "ok"]
    if not ok_entries:
        raise SystemExit(f"{args.camera}: no marker produced a usable Sim(3) scale.")

    scales = np.array([e["scale_m_per_slam_unit"] for e in ok_entries], dtype=float)
    weights = np.array([e["detections_kept"] for e in ok_entries], dtype=float)
    combined_median = float(np.median(scales))
    combined_weighted_mean = float(np.sum(scales * weights) / np.sum(weights))

    summary = {
        "camera": args.camera,
        "raw_csv": str(raw_csv),
        "camera_config": str(args.camera_config),
        "marker_length_m": args.marker_length_m,
        "markers_attempted": marker_ids,
        "per_marker": per_marker,
        "markers_used_for_combination": [e["marker_id"] for e in ok_entries],
        "recommended_scale_m_per_slam_unit": combined_median,
        "combination_method": "median across per-marker Sim(3) scales",
        "scale_weighted_mean_m_per_slam_unit": combined_weighted_mean,
        "scale_weighted_mean_method": "detections_kept-weighted mean, reported for comparison only",
    }
    out_path = out_dir / f"{args.camera}_sim3_scale_all_markers_summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"\n{args.camera}: {len(ok_entries)}/{len(marker_ids)} marker(s) produced a usable scale: "
          f"{[round(s, 4) for s in scales]}")
    print(f"Recommended scale (median across markers): {combined_median:.6f} m/slam-unit")
    print(f"(detections-weighted mean, for comparison: {combined_weighted_mean:.6f} m/slam-unit)")
    print(f"Output folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
