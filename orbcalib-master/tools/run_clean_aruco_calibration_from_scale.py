#!/usr/bin/env python3
"""Run trajectory extrinsic calibration from an existing clean scale recovery folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from clean_aruco_workflow import (
    append_marker_ids,
    append_optional,
    default_raw_csv,
    require_run_dir,
    run_logged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Run folder used for both cameras.")
    parser.add_argument("--camera1-run-dir", type=Path, help="Run folder for camera1. Defaults to --run-dir.")
    parser.add_argument("--camera2-run-dir", type=Path, help="Run folder for camera2. Defaults to --run-dir.")
    parser.add_argument("--camera1", required=True)
    parser.add_argument("--camera2", required=True)
    parser.add_argument("--camera1-config", required=True, type=Path)
    parser.add_argument("--camera2-config", required=True, type=Path)
    parser.add_argument("--camera1-raw-csv", type=Path)
    parser.add_argument("--camera2-raw-csv", type=Path)
    parser.add_argument("--scale-root", required=True, type=Path)
    parser.add_argument(
        "--scale-source",
        choices=("aruco-points", "ground", "sim3"),
        required=True,
        help="Recovered scale/alignment source to use for calibration.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override calibration output directory.")
    parser.add_argument("--marker-id", type=int, action="append", help="Force marker id. Can be repeated.")
    parser.add_argument("--marker-length-m", type=float, default=0.182)
    parser.add_argument("--dictionary-size", type=int, default=50)
    parser.add_argument("--keyframe-candidates", type=int, default=5)
    parser.add_argument("--visual-aruco-min-detections", type=int, default=4)
    parser.add_argument("--visual-aruco-max-rms-px", type=float, default=3.0)
    parser.add_argument("--visual-aruco-max-keyframes", type=int)
    parser.add_argument("--visual-aruco-trim-mad", type=float, default=3.5)
    parser.add_argument("--max-bracket-width-s", type=float)
    parser.add_argument("--max-distance-from-anchor-m", type=float)
    parser.add_argument("--camera1-max-distance-from-anchor-m", type=float)
    parser.add_argument("--camera2-max-distance-from-anchor-m", type=float)
    parser.add_argument("--max-time-from-anchor-s", type=float)
    parser.add_argument("--camera1-max-time-from-anchor-s", type=float)
    parser.add_argument("--camera2-max-time-from-anchor-s", type=float)
    parser.add_argument("--raw-ros-timestamps", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def aruco_points_summary(scale_root: Path, camera: str) -> Path:
    return scale_root / "aruco_points_scale" / camera / f"{camera}_aruco_scale_summary.json"


def aruco_points_csv(scale_root: Path, camera: str) -> Path:
    return scale_root / "aruco_points_scale" / camera / f"{camera}_aruco_scale_points.csv"


def load_aruco_points_scale(scale_root: Path, camera: str) -> float:
    with aruco_points_summary(scale_root, camera).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return float(summary["recommended_scale_m_per_slam_unit"])


def load_ground_scale(scale_root: Path, camera: str) -> float:
    summary_path = scale_root / "ground_scale" / "aruco_ground_scale_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    for item in summary.get("cameras", []):
        if item.get("camera") == camera:
            return float(item["recommended_scale_m_per_slam_unit"])
    raise RuntimeError(f"No ground scale for {camera} in {summary_path}")


def default_output_dir(args: argparse.Namespace, camera1_run_dir: Path, camera2_run_dir: Path) -> Path:
    names = {
        "aruco-points": "calibration_aruco_points_scale",
        "ground": "calibration_ground_scale",
        "sim3": "calibration_sim3_scale",
    }
    if args.run_dir is not None:
        root = args.run_dir
    elif camera1_run_dir == camera2_run_dir:
        root = camera1_run_dir
    else:
        root = args.scale_root.parent
    return root / names[args.scale_source]


def build_alignment_cmd(
    *,
    args: argparse.Namespace,
    camera1_raw: Path,
    camera2_raw: Path,
    output_dir: Path,
    alignment_source: str,
    camera1_scale: Optional[float] = None,
    camera2_scale: Optional[float] = None,
) -> list[str]:
    pair = f"{args.camera1}_{args.camera2}"
    align_dir = output_dir / "aruco_alignment"
    cmd = [
        "python3",
        "tools/align_trajectories_to_aruco.py",
        "--camera1-name",
        args.camera1,
        "--camera1-points-csv",
        str(aruco_points_csv(args.scale_root, args.camera1)),
        "--camera1-raw-csv",
        str(camera1_raw),
        "--camera1-config",
        str(args.camera1_config),
        "--camera2-name",
        args.camera2,
        "--camera2-points-csv",
        str(aruco_points_csv(args.scale_root, args.camera2)),
        "--camera2-raw-csv",
        str(camera2_raw),
        "--camera2-config",
        str(args.camera2_config),
        "--marker-length-m",
        str(args.marker_length_m),
        "--dictionary-size",
        str(args.dictionary_size),
        "--alignment-source",
        alignment_source,
        "--keyframe-candidates",
        str(args.keyframe_candidates),
        "--visual-aruco-min-detections",
        str(args.visual_aruco_min_detections),
        "--visual-aruco-max-rms-px",
        str(args.visual_aruco_max_rms_px),
        "--visual-aruco-trim-mad",
        str(args.visual_aruco_trim_mad),
        "--output-plot",
        str(align_dir / f"{pair}_trajectories.png"),
        "--output-alignment-json",
        str(align_dir / f"{pair}_alignment.json"),
        "--output-extrinsic-yaml",
        str(align_dir / f"{pair}_extrinsic.yaml"),
    ]
    append_marker_ids(cmd, args.marker_id)
    append_optional(cmd, "--visual-aruco-max-keyframes", args.visual_aruco_max_keyframes)
    append_optional(cmd, "--camera1-scale", camera1_scale)
    append_optional(cmd, "--camera2-scale", camera2_scale)
    append_optional(cmd, "--max-bracket-width-s", args.max_bracket_width_s)
    append_optional(cmd, "--max-distance-from-anchor-m", args.max_distance_from_anchor_m)
    append_optional(cmd, "--camera1-max-distance-from-anchor-m", args.camera1_max_distance_from_anchor_m)
    append_optional(cmd, "--camera2-max-distance-from-anchor-m", args.camera2_max_distance_from_anchor_m)
    append_optional(cmd, "--max-time-from-anchor-s", args.max_time_from_anchor_s)
    append_optional(cmd, "--camera1-max-time-from-anchor-s", args.camera1_max_time_from_anchor_s)
    append_optional(cmd, "--camera2-max-time-from-anchor-s", args.camera2_max_time_from_anchor_s)
    if args.raw_ros_timestamps:
        cmd.append("--raw-ros-timestamps")
    if args.show:
        cmd.append("--show")
    return cmd


def write_calibration_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    camera1_scale: Optional[float],
    camera2_scale: Optional[float],
    alignment_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, object] = {
        "scale_source": args.scale_source,
        "alignment_source": alignment_source,
        "scale_root": str(args.scale_root),
        "camera1": args.camera1,
        "camera2": args.camera2,
        "camera1_scale_m_per_slam_unit": camera1_scale,
        "camera2_scale_m_per_slam_unit": camera2_scale,
        "marker_policy": "forced " + ",".join(map(str, args.marker_id)) if args.marker_id else "best shared marker",
    }
    with (output_dir / "calibration_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    camera1_run_dir = require_run_dir(args, "camera1")
    camera2_run_dir = require_run_dir(args, "camera2")
    camera1_raw = args.camera1_raw_csv or default_raw_csv(camera1_run_dir, args.camera1)
    camera2_raw = args.camera2_raw_csv or default_raw_csv(camera2_run_dir, args.camera2)
    output_dir = args.output_dir or default_output_dir(args, camera1_run_dir, camera2_run_dir)

    camera1_scale: Optional[float] = None
    camera2_scale: Optional[float] = None
    if args.scale_source == "aruco-points":
        alignment_source = "anchor"
        camera1_scale = load_aruco_points_scale(args.scale_root, args.camera1)
        camera2_scale = load_aruco_points_scale(args.scale_root, args.camera2)
    elif args.scale_source == "ground":
        alignment_source = "anchor"
        camera1_scale = load_ground_scale(args.scale_root, args.camera1)
        camera2_scale = load_ground_scale(args.scale_root, args.camera2)
    else:
        alignment_source = "visual-aruco-sim3"

    write_calibration_manifest(
        args=args,
        output_dir=output_dir,
        camera1_scale=camera1_scale,
        camera2_scale=camera2_scale,
        alignment_source=alignment_source,
    )
    cmd = build_alignment_cmd(
        args=args,
        camera1_raw=camera1_raw,
        camera2_raw=camera2_raw,
        output_dir=output_dir,
        alignment_source=alignment_source,
        camera1_scale=camera1_scale,
        camera2_scale=camera2_scale,
    )
    run_logged(cmd, output_dir / "aruco_alignment" / f"{args.camera1}_{args.camera2}_runner.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
