#!/usr/bin/env python3
"""Shared helpers for the clean ArUco scale and calibration workflow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from estimate_aruco_ground_plane_scale import summarize_camera


DEFAULT_AGILEX_CAMERA_HEIGHTS_M = {
    "front": 0.46726490108354013,
    "front_aux": 0.46726490108354013,
    "back": 0.45869258774676547,
    "back_aux": 0.45869258774676547,
    "left": 0.536276885562841,
    "left_aux": 0.536276885562841,
    "right": 0.5447200312305958,
    "right_aux": 0.5447200312305958,
}


def require_run_dir(args: argparse.Namespace, which: str) -> Path:
    explicit = getattr(args, f"{which}_run_dir")
    if explicit is not None:
        return explicit
    if args.run_dir is not None:
        return args.run_dir
    raise ValueError(f"Provide --run-dir or --{which}-run-dir")


def default_raw_csv(run_dir: Path, camera: str) -> Path:
    return run_dir / f"{camera}_raw_keyframe_observations.csv"


def append_optional(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def append_marker_ids(cmd: List[str], marker_ids: Optional[List[int]]) -> None:
    for marker_id in marker_ids or []:
        cmd.extend(["--marker-id", str(marker_id)])


def run_logged(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nRunning:")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        rc = process.wait()
    if rc != 0:
        raise SystemExit(rc)


def load_marker_pair_scale(summary_path: Path) -> Optional[float]:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    value = summary.get("recommended_scale_m_per_slam_unit")
    return None if value is None else float(value)


def camera_height(args: argparse.Namespace, camera: str, specific: Optional[float]) -> Optional[float]:
    if specific is not None:
        return specific
    if args.camera_height_m is not None:
        return args.camera_height_m
    return DEFAULT_AGILEX_CAMERA_HEIGHTS_M.get(camera)


def extract_aruco_points(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    camera: str,
    config: Path,
    raw_csv: Path,
) -> Dict[str, object]:
    out_dir = args.output_root / "aruco_points_scale" / camera
    cmd = [
        sys.executable,
        "tools/estimate_aruco_slam_scale.py",
        "--run-dir",
        str(run_dir),
        "--camera",
        camera,
        "--observations-csv",
        str(raw_csv),
        "--camera-config",
        str(config),
        "--marker-length-m",
        str(args.marker_length_m),
        "--dictionary-size",
        str(args.dictionary_size),
        "--margin-px",
        str(args.margin_px),
        "--min-points",
        str(args.min_points),
        "--min-slam-distance",
        str(args.min_slam_distance),
        "--max-pairs-per-marker",
        str(args.max_pairs_per_marker),
        "--outlier-mad",
        str(args.outlier_mad),
        "--out-dir",
        str(out_dir),
    ]
    append_marker_ids(cmd, args.marker_id)
    append_optional(cmd, "--min-real-distance-m", args.min_real_distance_m)
    append_optional(cmd, "--max-keyframes", args.max_keyframes)
    if args.no_debug_images:
        cmd.append("--no-debug-images")
    run_logged(cmd, out_dir / "run.log")

    summary_path = out_dir / f"{camera}_aruco_scale_summary.json"
    return {
        "points_csv": out_dir / f"{camera}_aruco_scale_points.csv",
        "support_points_csv": None,
        "summary_json": summary_path,
        "scale": load_marker_pair_scale(summary_path),
        "raw_csv": raw_csv,
    }


def estimate_ground_plane_scale_from_points(
    *,
    camera: str,
    points_csv: Path,
    support_points_csv: Optional[Path],
    output_dir: Path,
    camera_height_m: Optional[float],
    min_points: int,
    marker_length_m: float,
    ransac_threshold: float,
    ransac_iterations: int,
    seed: int,
    outlier_mad: float,
    broad_ground_fallback_below: int,
    disable_broad_ground_fallback: bool,
    fallback_min_support_points: int,
    fallback_marker_inlier_fraction: float,
) -> Optional[float]:
    rows: List[Dict[str, object]] = []
    with points_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    support_rows: List[Dict[str, object]] = []
    if support_points_csv is not None and support_points_csv.exists():
        with support_points_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                support_rows.append(dict(row))

    if camera_height_m is None:
        print(f"{camera}: no camera height available; ground-plane scale unavailable")
        return None
    ground_dir = output_dir if output_dir.name == "ground_scale" else output_dir / "ground_scale"
    checked_keyframes = len({str(row.get("kf_id", "")) for row in rows if row.get("kf_id", "") != ""})
    ground_args = argparse.Namespace(
        min_points=min_points,
        ransac_threshold=ransac_threshold,
        ransac_iterations=ransac_iterations,
        seed=seed,
        outlier_mad=outlier_mad,
        marker_length_m=marker_length_m,
        broad_ground_fallback_below=broad_ground_fallback_below,
        disable_broad_ground_fallback=disable_broad_ground_fallback,
        fallback_min_support_points=fallback_min_support_points,
        fallback_marker_inlier_fraction=fallback_marker_inlier_fraction,
    )
    try:
        summary = summarize_camera(
            camera,
            rows,
            checked_keyframes=checked_keyframes,
            hit_keyframes=checked_keyframes,
            camera_height_m=camera_height_m,
            args=ground_args,
            out_dir=ground_dir,
            support_points=support_rows,
        )
    except RuntimeError as exc:
        print(f"{camera}: {exc}; ground-plane scale unavailable")
        return None
    summary_path = ground_dir / "aruco_ground_scale_summary.json"
    existing = {"cameras": []}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
    existing["cameras"] = [item for item in existing.get("cameras", []) if item.get("camera") != camera]
    existing["cameras"].append(summary)
    existing["scale_source"] = "ground-plane"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
        handle.write("\n")

    scale = summary["recommended_scale_m_per_slam_unit"]
    if scale is None:
        print(f"{camera}: camera-plane distances did not produce a usable ground-plane scale")
        return None
    print(
        f"{camera}: ground-plane scale = {scale} "
        f"({summary['marker_plane_inlier_points']}/{summary['unique_marker_points']} marker plane inliers, "
        f"{summary['unique_support_points']} support points, mode={summary['plane_fit_mode']}, "
        f"height={camera_height_m} m)"
    )
    return float(scale)


def estimate_ground_scale(
    *,
    args: argparse.Namespace,
    camera: str,
    points_csv: Path,
    support_points_csv: Optional[Path],
    height_m: Optional[float],
) -> Optional[float]:
    return estimate_ground_plane_scale_from_points(
        camera=camera,
        points_csv=points_csv,
        support_points_csv=support_points_csv,
        output_dir=args.output_root / "ground_scale",
        camera_height_m=height_m,
        min_points=args.min_points,
        marker_length_m=args.marker_length_m,
        ransac_threshold=args.ransac_threshold,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
        outlier_mad=args.outlier_mad,
        broad_ground_fallback_below=args.broad_ground_fallback_below,
        disable_broad_ground_fallback=not args.enable_broad_ground_fallback,
        fallback_min_support_points=args.fallback_min_support_points,
        fallback_marker_inlier_fraction=args.fallback_marker_inlier_fraction,
    )
