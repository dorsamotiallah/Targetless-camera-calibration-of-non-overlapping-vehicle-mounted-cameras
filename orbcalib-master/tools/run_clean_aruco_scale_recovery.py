#!/usr/bin/env python3
"""Recover ArUco trajectory-alignment scales without running extrinsic calibration.

This is the first half of the clean workflow. It assumes ORB-SLAM has already
been run and raw keyframe observation CSVs already exist.

Output structure:

  <output-root>/aruco_points_scale/<camera>/
  <output-root>/ground_scale/
  <output-root>/sim3_scale/
  <output-root>/scale_report.json
  <output-root>/scale_report.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from align_trajectories_to_aruco import (
    collect_visual_aruco_pose_observations,
    fit_visual_aruco_sim3,
    load_trajectory,
    read_points_csv,
)
from estimate_checkerboard_camera_height import read_calibration
from clean_aruco_workflow import (
    camera_height,
    default_raw_csv,
    estimate_ground_scale,
    extract_aruco_points,
    require_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="Run folder used for both cameras.")
    parser.add_argument("--camera1-run-dir", type=Path, help="Run folder for camera1. Defaults to --run-dir.")
    parser.add_argument("--camera2-run-dir", type=Path, help="Run folder for camera2. Defaults to --run-dir.")
    parser.add_argument("--camera1", required=True)
    parser.add_argument("--camera2", required=True)
    parser.add_argument("--camera1-config", required=True, type=Path)
    parser.add_argument("--camera2-config", required=True, type=Path)
    parser.add_argument("--camera1-raw-csv", type=Path)
    parser.add_argument("--camera2-raw-csv", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--scale-mode",
        choices=("all", "aruco-points", "ground", "sim3"),
        default="all",
        help="Which scale source(s) to recover. Default: all.",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        action="append",
        help="Restrict scale recovery to marker id. Can be repeated. If omitted, all markers are used.",
    )
    parser.add_argument("--marker-length-m", type=float, default=0.182)
    parser.add_argument("--dictionary-size", type=int, default=50)
    parser.add_argument(
        "--dictionary-bits",
        type=int,
        default=6,
        choices=(4, 5, 6, 7),
        help="ArUco marker grid size. Default: 6 for DICT_6X6_*.",
    )
    parser.add_argument("--margin-px", type=float, default=0.0)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-real-distance-m", type=float)
    parser.add_argument("--min-slam-distance", type=float, default=1e-7)
    parser.add_argument("--max-pairs-per-marker", type=int, default=5000)
    parser.add_argument("--outlier-mad", type=float, default=3.5)
    parser.add_argument("--max-keyframes", type=int)
    parser.add_argument("--no-debug-images", action="store_true")

    parser.add_argument("--camera-height-m", type=float, help="Fallback height for both cameras.")
    parser.add_argument("--camera1-height-m", type=float)
    parser.add_argument("--camera2-height-m", type=float)
    parser.add_argument("--ransac-threshold", type=float, default=0.03)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-broad-ground-fallback", action="store_true")
    parser.add_argument("--broad-ground-fallback-below", type=int, default=20)
    parser.add_argument("--fallback-min-support-points", type=int, default=50)
    parser.add_argument("--fallback-marker-inlier-fraction", type=float, default=1.0)

    parser.add_argument("--visual-aruco-min-detections", type=int, default=4)
    parser.add_argument("--visual-aruco-max-rms-px", type=float, default=3.0)
    parser.add_argument("--visual-aruco-max-keyframes", type=int)
    parser.add_argument("--visual-aruco-trim-mad", type=float, default=3.5)
    parser.add_argument("--raw-ros-timestamps", action="store_true")
    return parser.parse_args()


def marker_counts(points_csv: Path, allowed: Optional[List[int]]) -> Dict[int, int]:
    counts: Dict[int, set] = {}
    for row in read_points_csv(points_csv):
        marker_id = int(row["marker_id"])
        if allowed is not None and marker_id not in allowed:
            continue
        counts.setdefault(marker_id, set()).add(int(row["kf_id"]))
    return {marker_id: len(kfs) for marker_id, kfs in counts.items()}


def choose_best_shared_marker(points1: Path, points2: Path, allowed: Optional[List[int]]) -> int:
    counts1 = marker_counts(points1, allowed)
    counts2 = marker_counts(points2, allowed)
    shared = sorted(set(counts1) & set(counts2))
    if not shared:
        raise RuntimeError("No shared marker id found in the extracted ArUco point CSVs.")
    return max(shared, key=lambda marker_id: counts1[marker_id] + counts2[marker_id])


def recover_sim3_scale(
    *,
    args: argparse.Namespace,
    camera: str,
    raw_csv: Path,
    config: Path,
    marker_id: int,
) -> Dict[str, object]:
    out_dir = args.output_root / "sim3_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = read_calibration(config)
    use_source_timestamps = not args.raw_ros_timestamps
    trajectory, timestamp_source = load_trajectory(raw_csv, camera, use_source_timestamps)
    observations = collect_visual_aruco_pose_observations(
        camera,
        raw_csv,
        marker_id,
        calib,
        args.marker_length_m,
        args.dictionary_size,
        args.dictionary_bits,
        trajectory,
        args.visual_aruco_max_rms_px,
        args.visual_aruco_max_keyframes,
    )
    scale, R, t, keep, diagnostics = fit_visual_aruco_sim3(
        camera,
        observations,
        args.visual_aruco_min_detections,
        args.visual_aruco_trim_mad,
    )
    summary = {
        "camera": camera,
        "raw_csv": str(raw_csv),
        "camera_config": str(config),
        "marker_id": marker_id,
        "scale_source": "visual_aruco_sim3",
        "recommended_scale_m_per_slam_unit": scale,
        "trajectory_timestamp_source": timestamp_source,
        "visual_marker_detections": len(observations),
        "visual_marker_detections_used": int(np.count_nonzero(keep)),
        "rotation_slam_to_marker": R.tolist(),
        "translation_slam_to_marker_m": t.tolist(),
        "diagnostics": diagnostics,
    }
    path = out_dir / f"{camera}_sim3_scale_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    summary["summary_json"] = str(path)
    return summary


def write_report(report: Dict[str, object], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "scale_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    lines = ["ArUco scale recovery report", ""]
    lines.append(f"camera pair: {report['camera1']} / {report['camera2']}")
    lines.append(f"marker policy: {report['marker_policy']}")
    lines.append("")
    for mode, values in report["scales"].items():
        lines.append(f"[{mode}]")
        if not values:
            lines.append("  not run")
        for camera, item in values.items():
            scale = item.get("recommended_scale_m_per_slam_unit")
            marker = item.get("marker_id", "all")
            lines.append(f"  {camera}: scale={scale} marker={marker} summary={item.get('summary_json')}")
        lines.append("")
    log_path = output_root / "scale_report.log"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved scale report: {json_path}")
    print(f"Saved scale log: {log_path}")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    camera1_run_dir = require_run_dir(args, "camera1")
    camera2_run_dir = require_run_dir(args, "camera2")
    camera1_raw = args.camera1_raw_csv or default_raw_csv(camera1_run_dir, args.camera1)
    camera2_raw = args.camera2_raw_csv or default_raw_csv(camera2_run_dir, args.camera2)

    camera1_info = extract_aruco_points(
        args=args,
        run_dir=camera1_run_dir,
        camera=args.camera1,
        config=args.camera1_config,
        raw_csv=camera1_raw,
    )
    camera2_info = extract_aruco_points(
        args=args,
        run_dir=camera2_run_dir,
        camera=args.camera2,
        config=args.camera2_config,
        raw_csv=camera2_raw,
    )

    report: Dict[str, object] = {
        "camera1": args.camera1,
        "camera2": args.camera2,
        "marker_policy": "restricted to " + ",".join(map(str, args.marker_id))
        if args.marker_id
        else "all markers for scale extraction; best shared marker for Sim3/alignment",
        "scales": {"aruco_points_scale": {}, "ground_scale": {}, "sim3_scale": {}},
    }

    if args.scale_mode in ("all", "aruco-points"):
        report["scales"]["aruco_points_scale"] = {
            args.camera1: {
                "recommended_scale_m_per_slam_unit": camera1_info["scale"],
                "summary_json": str(camera1_info["summary_json"]),
                "marker_id": args.marker_id or "all",
            },
            args.camera2: {
                "recommended_scale_m_per_slam_unit": camera2_info["scale"],
                "summary_json": str(camera2_info["summary_json"]),
                "marker_id": args.marker_id or "all",
            },
        }

    if args.scale_mode in ("all", "ground"):
        camera1_ground_scale = estimate_ground_scale(
            args=args,
            camera=args.camera1,
            points_csv=Path(camera1_info["points_csv"]),
            support_points_csv=None,
            height_m=camera_height(args, args.camera1, args.camera1_height_m),
        )
        camera2_ground_scale = estimate_ground_scale(
            args=args,
            camera=args.camera2,
            points_csv=Path(camera2_info["points_csv"]),
            support_points_csv=None,
            height_m=camera_height(args, args.camera2, args.camera2_height_m),
        )
        ground_summary = args.output_root / "ground_scale" / "aruco_ground_scale_summary.json"
        report["scales"]["ground_scale"] = {
            args.camera1: {
                "recommended_scale_m_per_slam_unit": camera1_ground_scale,
                "summary_json": str(ground_summary),
                "marker_id": args.marker_id or "all",
            },
            args.camera2: {
                "recommended_scale_m_per_slam_unit": camera2_ground_scale,
                "summary_json": str(ground_summary),
                "marker_id": args.marker_id or "all",
            },
        }

    if args.scale_mode in ("all", "sim3"):
        sim3_marker_id = choose_best_shared_marker(
            Path(camera1_info["points_csv"]),
            Path(camera2_info["points_csv"]),
            args.marker_id,
        )
        report["sim3_marker_id"] = sim3_marker_id
        camera1_sim3 = recover_sim3_scale(
            args=args,
            camera=args.camera1,
            raw_csv=camera1_raw,
            config=args.camera1_config,
            marker_id=sim3_marker_id,
        )
        camera2_sim3 = recover_sim3_scale(
            args=args,
            camera=args.camera2,
            raw_csv=camera2_raw,
            config=args.camera2_config,
            marker_id=sim3_marker_id,
        )
        report["scales"]["sim3_scale"] = {
            args.camera1: camera1_sim3,
            args.camera2: camera2_sim3,
        }

    write_report(report, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
