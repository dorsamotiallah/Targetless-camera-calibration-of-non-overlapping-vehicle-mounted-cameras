#!/usr/bin/env python3
"""Align two independent camera trajectories via ArUco, then calibrate their extrinsic.

Two stages, run together since the second reuses everything the first computes:

STAGE 1 -- alignment. Reads the per-point CSVs written by
estimate_aruco_slam_scale.py for two cameras, picks the marker id best
observed by both maps, and for each camera picks the keyframe with the most
marker points. At that keyframe it runs cv2.solvePnP once, against the real
marker geometry (object points = marker_x_m/marker_y_m/0, image points =
that keyframe's pixel detections), giving the camera's pose in the marker's
metric frame. The raw observations CSV (as written by the updated
atlas_export_observations) also carries every keyframe's own SLAM-frame pose
(camera_x/y/z plus a qw/qx/qy/qz world-to-camera rotation quaternion,
straight from ORB-SLAM3). Composing that anchor keyframe's SLAM-frame pose
with its marker-frame PnP pose gives the similarity transform (scale,
rotation, translation) from that camera's SLAM frame into the marker frame.
Scale (meters per SLAM unit) is normally extracted from marker point pairs
at the anchor keyframe, but can be supplied directly per camera instead via
--camera1-scale/--camera2-scale (e.g. from a ground-plane-fit scale
estimate), which skips that extraction. That single per-camera transform is
applied to every keyframe's full pose (position and orientation). Both
cameras' transformed trajectories are plotted together and saved to a
summary JSON.

STAGE 2 -- extrinsic calibration, reusing stage 1's two trajectories. Since
the two cameras share no visual content, keyframes can't be matched by
content -- only by time. For every camera2 keyframe timestamp, camera1's
pose is interpolated (linear position, SLERP rotation) between its two
bracketing keyframes. Every interpolated match gives one estimate of the
camera2->camera1 extrinsic; since the rig is rigid these should all agree,
so they are combined with a robust rotation/translation average. Distance
from each camera's own anchor keyframe is the dominant quality signal
(monocular SLAM scale/pose drift grows with distance from it), so matches
are filtered by that, and a sensitivity table is always printed so the
cutoff can be chosen deliberately rather than guessed.

STAGE 3 -- joint optimization refinement, reusing stage 2's surviving
matches. The averaged (chordal-mean rotation, median translation) result
above is a "solve each match independently, then reduce" estimate. This
stage instead fits a single (R, t) directly against every surviving match's
raw pose pair at once (a joint nonlinear least-squares over SE(3), seeded
from the averaged result), with a robust (Huber) loss so remaining outlier
matches are automatically downweighted rather than relying solely on the
hard anchor-distance/time cutoffs. Both the averaged and the optimized
result are printed and saved side by side so they can be compared run to
run -- the optimizer is not assumed to be better a priori.

Example:

  python3 tools/align_trajectories_to_aruco.py \\
    --camera1-name front \\
    --camera1-points-csv results_agilex/<run>/aruco_scale/front/front_aruco_scale_points.csv \\
    --camera1-raw-csv results_agilex/<run>/front_raw_keyframe_observations.csv \\
    --camera1-config config/sim/agilex_front_defished_cam.yaml \\
    --camera2-name left \\
    --camera2-points-csv results_agilex/<run>/aruco_scale/left/left_aruco_scale_points.csv \\
    --camera2-raw-csv results_agilex/<run>/left_raw_keyframe_observations.csv \\
    --camera2-config config/sim/agilex_left_defished_cam.yaml \\
    --max-distance-from-anchor-m 1.0
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize

from estimate_checkerboard_camera_height import CameraCalibration, read_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera1-name", required=True)
    parser.add_argument("--camera1-points-csv", required=True, type=Path, help="<camera>_aruco_scale_points.csv")
    parser.add_argument("--camera1-raw-csv", required=True, type=Path, help="<camera>_raw_keyframe_observations.csv")
    parser.add_argument("--camera1-config", required=True, type=Path, help="ORB-SLAM style camera intrinsics yaml")
    parser.add_argument(
        "--camera1-scale",
        type=float,
        help="camera1's SLAM map scale in meters per SLAM unit (e.g. from "
        "estimate_aruco_ground_plane_scale.py or a MATLAB ground-plane fit). If omitted, "
        "falls back to extracting it from marker point pairs at the anchor keyframe.",
    )

    parser.add_argument("--camera2-name", required=True)
    parser.add_argument("--camera2-points-csv", required=True, type=Path)
    parser.add_argument("--camera2-raw-csv", required=True, type=Path)
    parser.add_argument("--camera2-config", required=True, type=Path)
    parser.add_argument(
        "--camera2-scale",
        type=float,
        help="camera2's SLAM map scale in meters per SLAM unit. Same fallback as --camera1-scale.",
    )

    parser.add_argument(
        "--marker-id",
        type=int,
        action="append",
        help=(
            "Restrict marker selection to this id. Can be repeated. If omitted, the best shared "
            "marker is selected from all detected markers."
        ),
    )
    parser.add_argument(
        "--alignment-source",
        choices=("visual-aruco-sim3", "visual-aruco-motion-scale", "optimized-anchor"),
        required=True,
        help="How to align each SLAM map to the marker frame. 'visual-aruco-sim3' detects the marker "
        "in many keyframes, solves metric PnP for each, and fits a robust Sim(3) from SLAM camera "
        "centers to visual ArUco camera centers -- this both fits rotation and recovers its own scale. "
        "'visual-aruco-motion-scale' uses the same visual detections, recovers scale from robust "
        "pairwise camera-motion ratios, then fits rotation/translation. 'optimized-anchor' detects the "
        "marker in many keyframes like visual-aruco-sim3 and fits the same robust (MAD-trimmed, "
        "chordal-mean) rotation from all of them, but takes the metric scale from a required "
        "--camera1-scale/--camera2-scale value (e.g. a ground-plane or point-pair scale recovery) "
        "instead of fitting its own scale -- use this for any externally-supplied scale source. There "
        "is no single-anchor-keyframe option: a single keyframe's PnP solve has no protection against "
        "outliers (e.g. the planar-marker pose ambiguity) and was superseded by 'optimized-anchor', "
        "which is exactly as cheap to supply scale to but robust to any one bad keyframe.",
    )
    parser.add_argument(
        "--dictionary-size",
        type=int,
        default=50,
        help="OpenCV ArUco dictionary size used by the visual fallback. Default: 50.",
    )
    parser.add_argument(
        "--dictionary-bits",
        type=int,
        default=6,
        choices=(4, 5, 6, 7),
        help="ArUco marker grid size used by the visual fallback. Default: 6 for DICT_6X6_*.",
    )
    parser.add_argument(
        "--visual-aruco-min-detections",
        type=int,
        default=4,
        help="Minimum visual ArUco keyframe detections needed for --alignment-source visual-aruco-sim3. Default: 4.",
    )
    parser.add_argument(
        "--visual-aruco-max-rms-px",
        type=float,
        default=3.0,
        help="Reject visual ArUco PnP detections above this reprojection RMS in pixels. Default: 3.0.",
    )
    parser.add_argument(
        "--visual-aruco-max-keyframes",
        type=int,
        help="Inspect at most this many keyframes per camera for visual ArUco Sim(3). Default: all.",
    )
    parser.add_argument(
        "--visual-aruco-trim-mad",
        type=float,
        default=3.5,
        help="MAD cutoff for trimming visual ArUco Sim(3) residual outliers. 0 disables trimming. Default: 3.5.",
    )
    parser.add_argument(
        "--visual-aruco-motion-min-distance-m",
        type=float,
        default=0.05,
        help="Minimum ArUco-PnP camera-center displacement for pairwise motion-scale ratios. Default: 0.05 m.",
    )
    parser.add_argument(
        "--visual-aruco-motion-max-pairs",
        type=int,
        default=20000,
        help="Maximum visual ArUco detection pairs sampled for motion-scale recovery. 0 means all pairs. Default: 20000.",
    )
    parser.add_argument("--marker-length-m", type=float, default=0.182, help="Marker side length, for the plot outline only.")

    parser.add_argument(
        "--max-bracket-width-s",
        type=float,
        help="Drop matches where camera1's bracketing keyframes are farther apart than this "
        "many seconds (i.e. camera2's timestamp falls in a sparse stretch of camera1's "
        "trajectory). Default: no filtering, but a sensitivity table is always printed.",
    )
    parser.add_argument(
        "--max-distance-from-anchor-m",
        type=float,
        help="Drop matches where either camera's point is farther than this many meters (Euclidean, "
        "in the marker frame) from that camera's own anchor keyframe. Each camera's Sim3 is only "
        "well-constrained near its marker anchor -- monocular SLAM scale/pose drift grows with "
        "distance from it. Default: no filtering, but a sensitivity table is always printed. "
        "Euclidean distance is an imperfect proxy for accumulated drift (e.g. a keyframe can "
        "revisit the anchor's physical location much later, after more drift, and still look "
        "close by this measure) -- see also --max-time-from-anchor-s.",
    )
    parser.add_argument(
        "--camera1-max-distance-from-anchor-m",
        type=float,
        help="Camera1-specific anchor distance cutoff in meters. Overrides --max-distance-from-anchor-m for camera1.",
    )
    parser.add_argument(
        "--camera2-max-distance-from-anchor-m",
        type=float,
        help="Camera2-specific anchor distance cutoff in meters. Overrides --max-distance-from-anchor-m for camera2.",
    )
    parser.add_argument(
        "--max-time-from-anchor-s",
        type=float,
        help="Drop matches where either camera's keyframe is farther than this many seconds from "
        "that camera's own anchor keyframe, independent of --max-distance-from-anchor-m. SLAM "
        "drift accumulates with trajectory distance, not physical proximity, so a keyframe can be "
        "Euclidean-close to the anchor while being far away in time (e.g. a later revisit of the "
        "same spot) and still carry much more accumulated drift than the distance filter alone "
        "would catch. Default: no filtering, but a sensitivity table is always printed.",
    )
    parser.add_argument(
        "--camera1-max-time-from-anchor-s",
        type=float,
        help="Camera1-specific anchor time cutoff in seconds. Overrides --max-time-from-anchor-s for camera1.",
    )
    parser.add_argument(
        "--camera2-max-time-from-anchor-s",
        type=float,
        help="Camera2-specific anchor time cutoff in seconds. Overrides --max-time-from-anchor-s for camera2.",
    )

    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Path to save the 3D trajectory plot (PNG). Defaults to "
        "<run-dir>/aruco_alignment/<camera1>_<camera2>_trajectories.png, where <run-dir> is "
        "--camera1-raw-csv's parent directory.",
    )
    parser.add_argument(
        "--output-alignment-json",
        type=Path,
        help="Path to save the stage-1 alignment summary JSON. Defaults to "
        "<run-dir>/aruco_alignment/<camera1>_<camera2>_alignment.json.",
    )
    parser.add_argument(
        "--output-extrinsic-yaml",
        type=Path,
        help="Path to save the stage-2 extrinsic result (T_<camera1>_<camera2>, matching the "
        "robot_relative_extrinsics.yaml ground-truth format). Defaults to "
        "<run-dir>/aruco_alignment/<camera1>_<camera2>_extrinsic.yaml.",
    )
    parser.add_argument("--show", action="store_true", help="Also show the plot interactively (it is always saved).")
    parser.add_argument(
        "--hide-anchor-marker",
        action="store_true",
        help="Don't draw the single starred anchor/reference keyframe in the trajectory plots. The "
        "SLAM-to-marker transform is derived from many keyframes (not one), so a single starred point "
        "would misleadingly suggest otherwise. The plot then shows just the trajectories and the marker.",
    )
    parser.add_argument(
        "--raw-ros-timestamps",
        action="store_true",
        help="Use the raw CSV timestamp column directly. By default, when manifest.txt and "
        "frame_pairs.csv are available beside a raw CSV, keyframe times are converted back "
        "to original source PNG timestamps so independently replayed runs can be aligned.",
    )

    parser.add_argument(
        "--optimize-loss",
        default="huber",
        choices=["linear", "huber", "soft_l1", "cauchy"],
        help="Robust loss for the stage-3 joint SE(3) optimization refinement. 'linear' is a "
        "plain (non-robust) least-squares fit. Default: huber.",
    )
    parser.add_argument(
        "--optimize-f-scale-m",
        type=float,
        help="Robust loss transition scale (meters-equivalent residual magnitude) for stage 3 -- "
        "residuals below this are treated as inliers (quadratic cost), larger ones are "
        "downweighted. Default: 1.5x the median translation deviation from the averaged result.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Stage 1: alignment
# --------------------------------------------------------------------------------------


def read_points_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"marker_id", "kf_id", "u", "v", "marker_x_m", "marker_y_m", "mp_x", "mp_y", "mp_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append(
                {
                    "marker_id": int(float(row["marker_id"])),
                    "kf_id": int(float(row["kf_id"])),
                    "u": float(row["u"]),
                    "v": float(row["v"]),
                    "marker_x_m": float(row["marker_x_m"]),
                    "marker_y_m": float(row["marker_y_m"]),
                    "mp_x": float(row["mp_x"]),
                    "mp_y": float(row["mp_y"]),
                    "mp_z": float(row["mp_z"]),
                }
            )
    return rows


def marker_corner_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def keyframe_counts_by_marker(rows: List[Dict[str, float]]) -> Dict[int, int]:
    marker_to_kfs: Dict[int, set] = {}
    for row in rows:
        marker_to_kfs.setdefault(row["marker_id"], set()).add(row["kf_id"])
    return {marker_id: len(kfs) for marker_id, kfs in marker_to_kfs.items()}


def choose_marker_id(
    rows1: List[Dict[str, float]],
    rows2: List[Dict[str, float]],
    name1: str,
    name2: str,
    allowed_marker_ids: Optional[List[int]],
) -> int:
    counts1 = keyframe_counts_by_marker(rows1)
    counts2 = keyframe_counts_by_marker(rows2)
    all_markers = sorted(set(counts1) | set(counts2))
    allowed = set(allowed_marker_ids or [])

    print(f"{'marker_id':>9}  {name1 + '_kfs':>12}  {name2 + '_kfs':>12}")
    for marker_id in all_markers:
        print(f"{marker_id:9d}  {counts1.get(marker_id, 0):12d}  {counts2.get(marker_id, 0):12d}")

    candidate_markers = [m for m in all_markers if not allowed or m in allowed]
    if allowed and not candidate_markers:
        requested = ", ".join(map(str, sorted(allowed)))
        raise SystemExit(f"Requested marker id(s) {requested} were not found in either input.")

    shared = [m for m in candidate_markers if counts1.get(m, 0) > 0 and counts2.get(m, 0) > 0]
    if not shared:
        if allowed:
            requested = ", ".join(map(str, sorted(allowed)))
            if len(allowed) == 1:
                marker_id = next(iter(allowed))
                print(
                    f"marker_id {marker_id} does not have SLAM marker points in both cameras; "
                    "direct visual ArUco fallback will be tried where needed."
                )
                return marker_id
            raise SystemExit(f"No requested marker id is observed by both {name1} and {name2}: {requested}")
        raise SystemExit(f"No marker id is observed by both {name1} and {name2}")
    best = max(shared, key=lambda m: counts1[m] + counts2[m])
    policy = "allowed" if allowed else "all"
    print(f"Selected marker_id={best} from {policy} marker(s) ({name1}: {counts1[best]} kfs, {name2}: {counts2[best]} kfs)")
    return best


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    calib: CameraCalibration,
    prefer_planar: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if calib.model == "KannalaBrandt8":
        solve_image_points = cv2.fisheye.undistortPoints(image_points, calib.K, calib.D)
        solve_K = np.eye(3, dtype=np.float64)
        solve_D = None
    else:
        solve_image_points = image_points
        solve_K = calib.K
        solve_D = calib.D
    flags_to_try = [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_EPNP] if prefer_planar else [cv2.SOLVEPNP_EPNP]

    last_error: Optional[Exception] = None
    for flags in flags_to_try:
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, solve_image_points, solve_K, solve_D, flags=flags)
        except cv2.error as exc:  # some flags reject degenerate point configs
            last_error = exc
            continue
        if not ok:
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, solve_image_points, solve_K, solve_D, rvec, tvec)
        except cv2.error:
            pass
        R, _ = cv2.Rodrigues(rvec)
        if calib.model == "KannalaBrandt8":
            projected, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, calib.K, calib.D)
        else:
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, calib.K, calib.D)
        residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        rms_px = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        return R, tvec.reshape(3), rms_px

    raise RuntimeError(f"solvePnP failed for all flags tried (last error: {last_error})")


def aruco_dictionary(dictionary_size: int, dictionary_bits: int = 6) -> object:
    if dictionary_size not in (50, 100, 250, 1000):
        raise ValueError(f"Unsupported ArUco dictionary size: {dictionary_size}")
    name = f"DICT_{dictionary_bits}X{dictionary_bits}_{dictionary_size}"
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_bits}x{dictionary_bits}_{dictionary_size}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(
    image: np.ndarray,
    dictionary_size: int,
    dictionary_bits: int = 6,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    dictionary = aruco_dictionary(dictionary_size, dictionary_bits)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    return corners, ids


def resolve_container_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    prefixes = [
        ("/ws/src/orbcalib-master", repo_root),
        ("/ws/src/T7", Path("/media/civit/T7")),
        ("/ws/src/Agilex_Recordings", repo_root.parent / "Agilex Recordings"),
        ("/ws/src/NMC3D", repo_root.parent / "NMC3D"),
    ]
    for prefix, host_prefix in prefixes:
        if path_text.startswith(prefix):
            candidate = host_prefix / path_text[len(prefix) :].lstrip("/")
            if candidate.exists():
                return candidate
    return path


def sorted_pngs(folder: Path) -> List[Path]:
    frames = sorted(folder.glob("*.png"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {folder}")
    return frames


def load_keyframe_rows(raw_csv_path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with raw_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"kf_id", "frame_id", "timestamp"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{raw_csv_path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            kf_id = int(float(row["kf_id"]))
            rows.setdefault(kf_id, row)
    return rows


def image_for_keyframe(raw_csv_path: Path, camera_name: str, raw_row: Dict[str, str]) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = raw_csv_path.parent
    manifest = read_manifest(run_dir / "manifest.txt")
    side = camera_side_from_manifest(camera_name, manifest)
    timestamp = float(raw_row["timestamp"])

    frame_pairs_csv = run_dir / "frame_pairs.csv"
    if frame_pairs_csv.exists() and side is not None:
        ros_key = f"camera{side}_ros_stamp"
        png_key = f"camera{side}_png"
        best: Optional[Tuple[float, Path]] = None
        with frame_pairs_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if {ros_key, png_key}.issubset(set(reader.fieldnames or [])):
                for row in reader:
                    distance = abs(float(row[ros_key]) - timestamp)
                    path = resolve_container_path(row[png_key], repo_root)
                    if best is None or distance < best[0]:
                        best = (distance, path)
        if best is not None:
            return best[1]

    if side is not None:
        dir_key = f"camera{side}_dir"
        if manifest.get(dir_key):
            frames = sorted_pngs(resolve_container_path(manifest[dir_key], repo_root))
            frame_id = int(float(raw_row["frame_id"]))
            if 0 <= frame_id < len(frames):
                return frames[frame_id]
            if 0 <= frame_id - 1 < len(frames):
                return frames[frame_id - 1]
    raise ValueError(f"Could not resolve image for {camera_name} kf row from {raw_csv_path}")


def polygon_area(points: np.ndarray) -> float:
    pts = points.reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def read_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line or line.startswith(" "):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def camera_side_from_manifest(camera_name: str, manifest: Dict[str, str]) -> Optional[int]:
    normalized = camera_name.strip().lower()
    camera1 = manifest.get("camera1_name", "").strip().lower()
    camera2 = manifest.get("camera2_name", "").strip().lower()
    if normalized in {"camera1", "src"} or normalized == camera1:
        return 1
    if normalized in {"camera2", "dst"} or normalized == camera2:
        return 2
    return None


def load_source_timestamp_map(run_dir: Path, camera_name: str) -> Optional[List[Tuple[float, float]]]:
    """Return sorted (ros_timestamp_s, source_timestamp_s) entries for this camera.

    Raw keyframe CSV timestamps are ROS playback times. Those are only comparable
    within one replay process. The source PNG timestamp is the stable dataset time
    that remains comparable across independently replayed/single-camera runs.
    """
    manifest = read_manifest(run_dir / "manifest.txt")
    side = camera_side_from_manifest(camera_name, manifest)
    if side is None:
        return None

    frame_pairs_csv = run_dir / "frame_pairs.csv"
    if not frame_pairs_csv.exists():
        manifest_value = manifest.get("frame_pairs_csv")
        if manifest_value:
            candidate = Path(manifest_value)
            if not candidate.is_absolute():
                # Manifest paths are normally relative to the repository root; the
                # run directory's parent is the results root, so resolve common
                # relative values conservatively from run_dir.parent.parent.
                candidate = run_dir.parent.parent / candidate
            if candidate.exists():
                frame_pairs_csv = candidate
    if not frame_pairs_csv.exists():
        return None

    ros_key = f"camera{side}_ros_stamp"
    source_key = f"camera{side}_source_stamp_ns"
    mapping: List[Tuple[float, float]] = []
    with frame_pairs_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if {ros_key, source_key} - fields:
            return None
        for row in reader:
            mapping.append((float(row[ros_key]), float(row[source_key]) * 1e-9))
    mapping.sort(key=lambda item: item[0])
    return mapping or None


def nearest_source_timestamp(mapping: List[Tuple[float, float]], ros_timestamp: float) -> float:
    ros_times = [item[0] for item in mapping]
    idx = bisect.bisect_left(ros_times, ros_timestamp)
    candidates = []
    if idx < len(mapping):
        candidates.append(mapping[idx])
    if idx > 0:
        candidates.append(mapping[idx - 1])
    if not candidates:
        return ros_timestamp
    return min(candidates, key=lambda item: abs(item[0] - ros_timestamp))[1]


def load_trajectory(
    path: Path,
    camera_name: str,
    use_source_timestamps: bool = True,
) -> Tuple[Dict[int, Tuple[float, np.ndarray, np.ndarray]], str]:
    """Single pass over the raw observations CSV: kf_id -> (timestamp, camera_center, R_cw).

    R_cw is the keyframe's own SLAM-frame world-to-camera rotation
    (p_cam = R_cw @ p_world + t_cw), straight from ORB-SLAM3's KeyFrame::GetRotation()
    via the qw/qx/qy/qz columns written by atlas_export_observations.
    """
    trajectory: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
    timestamp_map = load_source_timestamp_map(path.parent, camera_name) if use_source_timestamps else None
    timestamp_source = "source_png" if timestamp_map is not None else "raw_ros"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"kf_id", "timestamp", "camera_x", "camera_y", "camera_z", "qw", "qx", "qy", "qz"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}. "
                "Re-export it with the updated atlas_export_observations."
            )
        for row in reader:
            kf_id = int(row["kf_id"])
            if kf_id in trajectory:
                continue
            center = np.array([float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])])
            q = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
            raw_timestamp = float(row["timestamp"])
            timestamp = nearest_source_timestamp(timestamp_map, raw_timestamp) if timestamp_map is not None else raw_timestamp
            trajectory[kf_id] = (timestamp, center, quaternion_to_matrix(q))
    return trajectory, timestamp_source


def collect_visual_aruco_pose_observations(
    name: str,
    raw_csv_path: Path,
    marker_id: int,
    calib: CameraCalibration,
    marker_length_m: float,
    dictionary_size: int,
    dictionary_bits: int,
    trajectory: Dict[int, Tuple[float, np.ndarray, np.ndarray]],
    max_rms_px: float,
    max_keyframes: Optional[int],
) -> List[Dict[str, object]]:
    keyframes = load_keyframe_rows(raw_csv_path)
    object_points = marker_corner_object_points(marker_length_m)
    observations: List[Dict[str, object]] = []
    checked_keyframes = 0
    detected_keyframes = 0
    rejected_rms = 0
    missing_images = 0

    for kf_id, raw_row in sorted(keyframes.items()):
        if max_keyframes is not None and checked_keyframes >= max_keyframes:
            break
        if kf_id not in trajectory:
            continue
        checked_keyframes += 1
        try:
            image_path = image_for_keyframe(raw_csv_path, name, raw_row)
        except (ValueError, FileNotFoundError):
            missing_images += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing_images += 1
            continue
        corners_list, ids = detect_markers(image, dictionary_size, dictionary_bits)
        if ids is None:
            continue

        best: Optional[Dict[str, object]] = None
        for corners, detected_id in zip(corners_list, ids.reshape(-1)):
            if int(detected_id) != marker_id:
                continue
            corners_2d = corners.reshape(4, 2).astype(np.float64)
            try:
                R_marker_to_camera, t_marker_to_camera, rms_px = solve_pnp(
                    object_points,
                    corners_2d,
                    calib,
                    prefer_planar=True,
                )
            except RuntimeError:
                continue
            if rms_px > max_rms_px:
                rejected_rms += 1
                continue
            _, center_slam, R_slam_to_camera = trajectory[kf_id]
            center_marker = -R_marker_to_camera.T @ t_marker_to_camera
            R_slam_to_marker = R_marker_to_camera.T @ R_slam_to_camera
            area_px2 = polygon_area(corners_2d)
            candidate = {
                "kf_id": kf_id,
                "timestamp": trajectory[kf_id][0],
                "image_path": str(image_path),
                "rms_px": rms_px,
                "area_px2": area_px2,
                "center_slam": center_slam,
                "center_marker": center_marker,
                "R_slam_to_marker": R_slam_to_marker,
            }
            if best is None or (rms_px, -area_px2) < (float(best["rms_px"]), -float(best["area_px2"])):
                best = candidate

        if best is not None:
            detected_keyframes += 1
            observations.append(best)

    print(
        f"{name}: visual ArUco Sim(3) checked {checked_keyframes} keyframe(s), "
        f"kept {len(observations)} detection(s), rejected {rejected_rms} by RMS>{max_rms_px:g}px"
        + (f", {missing_images} image(s) unavailable" if missing_images else "")
    )
    if observations:
        rms_values = np.array([float(obs["rms_px"]) for obs in observations], dtype=np.float64)
        areas = np.array([float(obs["area_px2"]) for obs in observations], dtype=np.float64)
        print(
            f"{name}: visual ArUco RMS px median={np.median(rms_values):.3f}, "
            f"max={rms_values.max():.3f}; area px^2 median={np.median(areas):.1f}"
        )
    return observations


def fit_visual_aruco_sim3(
    name: str,
    observations: List[Dict[str, object]],
    min_detections: int,
    trim_mad: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(observations) < min_detections:
        raise ValueError(
            f"{name}: only {len(observations)} visual ArUco detection(s), need {min_detections} "
            "for visual Sim(3) scale recovery"
        )

    centers_slam = np.array([obs["center_slam"] for obs in observations], dtype=np.float64)
    centers_marker = np.array([obs["center_marker"] for obs in observations], dtype=np.float64)
    rotations = np.array([obs["R_slam_to_marker"] for obs in observations], dtype=np.float64)
    keep = np.ones(len(observations), dtype=bool)

    R_slam_to_marker = np.eye(3, dtype=np.float64)
    scale = 1.0
    t_slam_to_marker = np.zeros(3, dtype=np.float64)
    residuals = np.zeros(len(observations), dtype=np.float64)

    for _ in range(5):
        if int(np.count_nonzero(keep)) < min_detections:
            break
        R_slam_to_marker = rotation_chordal_mean(rotations[keep])
        slam_rotated = centers_slam[keep] @ R_slam_to_marker.T
        marker_kept = centers_marker[keep]
        slam_centered = slam_rotated - slam_rotated.mean(axis=0)
        marker_centered = marker_kept - marker_kept.mean(axis=0)
        denom = float(np.sum(slam_centered**2))
        if denom <= 1e-12:
            raise ValueError(f"{name}: visual ArUco detections have too little SLAM motion to recover scale")
        scale = float(np.sum(slam_centered * marker_centered) / denom)
        if scale <= 0.0:
            raise ValueError(f"{name}: visual ArUco Sim(3) produced non-positive scale {scale}")
        t_candidates = marker_kept - scale * slam_rotated
        t_slam_to_marker = np.median(t_candidates, axis=0)
        predicted_all = scale * (centers_slam @ R_slam_to_marker.T) + t_slam_to_marker
        residuals = np.linalg.norm(predicted_all - centers_marker, axis=1)

        if trim_mad <= 0.0:
            break
        kept_residuals = residuals[keep]
        median = float(np.median(kept_residuals))
        mad = float(np.median(np.abs(kept_residuals - median)))
        if mad <= 1e-12:
            break
        threshold = median + trim_mad * 1.4826 * mad
        new_keep = residuals <= threshold
        if int(np.count_nonzero(new_keep)) < min_detections or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    kept_residuals = residuals[keep]
    rotation_devs = np.array([rotation_angle_deg(R, R_slam_to_marker) for R in rotations], dtype=np.float64)
    diagnostics = {
        "visual_detections_total": int(len(observations)),
        "visual_detections_used": int(np.count_nonzero(keep)),
        "visual_detections_rejected": int(len(observations) - np.count_nonzero(keep)),
        "visual_sim3_residual_m": {
            "median_all": float(np.median(residuals)),
            "mean_all": float(np.mean(residuals)),
            "max_all": float(np.max(residuals)),
            "median_used": float(np.median(kept_residuals)),
            "mean_used": float(np.mean(kept_residuals)),
            "max_used": float(np.max(kept_residuals)),
        },
        "visual_rotation_deviation_deg": {
            "median": float(np.median(rotation_devs)),
            "mean": float(np.mean(rotation_devs)),
            "max": float(np.max(rotation_devs)),
        },
    }
    print(
        f"{name}: visual ArUco Sim(3) scale = {scale:.6f} m/slam-unit "
        f"using {diagnostics['visual_detections_used']}/{diagnostics['visual_detections_total']} detection(s); "
        f"residual median={diagnostics['visual_sim3_residual_m']['median_used']:.4f} m, "
        f"rotation dev median={diagnostics['visual_rotation_deviation_deg']['median']:.3f} deg"
    )
    return scale, R_slam_to_marker, t_slam_to_marker, keep, diagnostics


def pairwise_motion_scale_ratios(
    centers_slam: np.ndarray,
    centers_marker: np.ndarray,
    keep: np.ndarray,
    min_marker_distance_m: float,
    max_pairs: int,
) -> np.ndarray:
    indices = np.flatnonzero(keep)
    if len(indices) < 2:
        return np.empty((0,), dtype=np.float64)

    total_pairs = len(indices) * (len(indices) - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        pairs = [(int(indices[i]), int(indices[j])) for i in range(len(indices)) for j in range(i + 1, len(indices))]
    else:
        rng = np.random.default_rng(0)
        seen = set()
        pairs = []
        while len(pairs) < max_pairs:
            i, j = sorted(rng.choice(indices, size=2, replace=False).tolist())
            key = (int(i), int(j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)

    ratios: List[float] = []
    for i, j in pairs:
        marker_distance = float(np.linalg.norm(centers_marker[j] - centers_marker[i]))
        if marker_distance < min_marker_distance_m:
            continue
        slam_distance = float(np.linalg.norm(centers_slam[j] - centers_slam[i]))
        if slam_distance <= 1e-12:
            continue
        ratios.append(marker_distance / slam_distance)
    return np.asarray(ratios, dtype=np.float64)


def robust_median(values: np.ndarray, trim_mad: float) -> Tuple[float, np.ndarray, Dict[str, object]]:
    if values.size == 0:
        raise ValueError("no usable values for robust median")
    keep = np.ones(values.shape[0], dtype=bool)
    if trim_mad > 0.0 and values.size >= 4:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-12:
            threshold = trim_mad * 1.4826 * mad
            keep = np.abs(values - median) <= threshold
            if not np.any(keep):
                keep = np.ones(values.shape[0], dtype=bool)
    used = values[keep]
    diagnostics = {
        "count": int(values.size),
        "used_count": int(used.size),
        "rejected_count": int(values.size - used.size),
        "median_all": float(np.median(values)),
        "mean_all": float(np.mean(values)),
        "std_all": float(np.std(values)),
        "median_used": float(np.median(used)),
        "mean_used": float(np.mean(used)),
        "std_used": float(np.std(used)),
        "min_used": float(np.min(used)),
        "max_used": float(np.max(used)),
    }
    return float(np.median(used)), keep, diagnostics


def fit_visual_aruco_motion_scale(
    name: str,
    observations: List[Dict[str, object]],
    min_detections: int,
    trim_mad: float,
    min_marker_distance_m: float,
    max_pairs: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(observations) < min_detections:
        raise ValueError(
            f"{name}: only {len(observations)} visual ArUco detection(s), need {min_detections} "
            "for visual ArUco motion-scale recovery"
        )

    centers_slam = np.array([obs["center_slam"] for obs in observations], dtype=np.float64)
    centers_marker = np.array([obs["center_marker"] for obs in observations], dtype=np.float64)
    rotations = np.array([obs["R_slam_to_marker"] for obs in observations], dtype=np.float64)
    keep = np.ones(len(observations), dtype=bool)
    residuals = np.zeros(len(observations), dtype=np.float64)
    ratio_diagnostics: Dict[str, object] = {}

    scale = 1.0
    R_slam_to_marker = np.eye(3, dtype=np.float64)
    t_slam_to_marker = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        if int(np.count_nonzero(keep)) < min_detections:
            break
        ratios = pairwise_motion_scale_ratios(
            centers_slam,
            centers_marker,
            keep,
            min_marker_distance_m,
            max_pairs,
        )
        if ratios.size == 0:
            raise ValueError(
                f"{name}: no visual ArUco motion pairs with marker displacement >= {min_marker_distance_m:g} m"
            )
        scale, _, ratio_diagnostics = robust_median(ratios, trim_mad)
        if scale <= 0.0:
            raise ValueError(f"{name}: visual ArUco motion-scale produced non-positive scale {scale}")

        R_slam_to_marker = rotation_chordal_mean(rotations[keep])
        slam_rotated = centers_slam @ R_slam_to_marker.T
        t_candidates = centers_marker[keep] - scale * slam_rotated[keep]
        t_slam_to_marker = np.median(t_candidates, axis=0)
        predicted = scale * slam_rotated + t_slam_to_marker
        residuals = np.linalg.norm(predicted - centers_marker, axis=1)

        if trim_mad <= 0.0:
            break
        kept_residuals = residuals[keep]
        median = float(np.median(kept_residuals))
        mad = float(np.median(np.abs(kept_residuals - median)))
        if mad <= 1e-12:
            break
        threshold = median + trim_mad * 1.4826 * mad
        new_keep = residuals <= threshold
        if int(np.count_nonzero(new_keep)) < min_detections or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    kept_residuals = residuals[keep]
    rotation_devs = np.array([rotation_angle_deg(R, R_slam_to_marker) for R in rotations], dtype=np.float64)
    diagnostics = {
        "visual_detections_total": int(len(observations)),
        "visual_detections_used": int(np.count_nonzero(keep)),
        "visual_detections_rejected": int(len(observations) - np.count_nonzero(keep)),
        "motion_pair_min_marker_distance_m": float(min_marker_distance_m),
        "motion_pair_scale": ratio_diagnostics,
        "visual_motion_scale_residual_m": {
            "median_all": float(np.median(residuals)),
            "mean_all": float(np.mean(residuals)),
            "max_all": float(np.max(residuals)),
            "median_used": float(np.median(kept_residuals)),
            "mean_used": float(np.mean(kept_residuals)),
            "max_used": float(np.max(kept_residuals)),
        },
        "visual_rotation_deviation_deg": {
            "median": float(np.median(rotation_devs)),
            "mean": float(np.mean(rotation_devs)),
            "max": float(np.max(rotation_devs)),
        },
    }
    print(
        f"{name}: visual ArUco motion scale = {scale:.6f} m/slam-unit "
        f"using {diagnostics['visual_detections_used']}/{diagnostics['visual_detections_total']} detection(s), "
        f"{ratio_diagnostics.get('used_count', 0)}/{ratio_diagnostics.get('count', 0)} motion pair(s); "
        f"residual median={diagnostics['visual_motion_scale_residual_m']['median_used']:.4f} m, "
        f"rotation dev median={diagnostics['visual_rotation_deviation_deg']['median']:.3f} deg"
    )
    return scale, R_slam_to_marker, t_slam_to_marker, keep, diagnostics


def align_camera_visual_aruco_sim3(
    name: str,
    marker_id: int,
    raw_csv_path: Path,
    calib: CameraCalibration,
    marker_length_m: float,
    dictionary_size: int,
    dictionary_bits: int,
    min_detections: int,
    max_rms_px: float,
    max_keyframes: Optional[int],
    trim_mad: float,
    use_source_timestamps: bool = True,
) -> Dict[str, object]:
    trajectory, timestamp_source = load_trajectory(raw_csv_path, name, use_source_timestamps)
    print(f"{name}: trajectory timestamps = {timestamp_source}")
    observations = collect_visual_aruco_pose_observations(
        name,
        raw_csv_path,
        marker_id,
        calib,
        marker_length_m,
        dictionary_size,
        dictionary_bits,
        trajectory,
        max_rms_px,
        max_keyframes,
    )
    scale, R_slam_to_marker, t_slam_to_marker, keep, diagnostics = fit_visual_aruco_sim3(
        name,
        observations,
        min_detections,
        trim_mad,
    )

    best_index = min(range(len(observations)), key=lambda idx: (float(observations[idx]["rms_px"]), -float(observations[idx]["area_px2"])))
    anchor_obs = observations[best_index]
    kf_id = int(anchor_obs["kf_id"])
    anchor_timestamp = float(anchor_obs["timestamp"])

    items = sorted(trajectory.items(), key=lambda kv: kv[1][0])
    kf_ids_sorted = np.array([k for k, _ in items], dtype=np.int64)
    timestamps = np.array([v[0] for _, v in items], dtype=np.float64)
    centers_slam = np.array([v[1] for _, v in items], dtype=np.float64)
    rotations_slam = np.array([v[2] for _, v in items], dtype=np.float64)

    positions_marker = scale * (centers_slam @ R_slam_to_marker.T) + t_slam_to_marker
    rotations_marker = rotations_slam @ R_slam_to_marker.T
    anchor_point = scale * (np.asarray(anchor_obs["center_slam"], dtype=np.float64) @ R_slam_to_marker.T) + t_slam_to_marker

    return {
        "name": name,
        "marker_id": marker_id,
        "kf_id": kf_id,
        "scale": scale,
        "scale_source": "visual_aruco_sim3",
        "scale_recovered": True,
        "scale_warning": None,
        "R_slam_to_marker": R_slam_to_marker,
        "t_slam_to_marker": t_slam_to_marker,
        "trajectory_kf_ids": kf_ids_sorted,
        "trajectory_timestamps": timestamps,
        "trajectory_timestamp_source": timestamp_source,
        "trajectory": positions_marker,
        "trajectory_rotations": rotations_marker,
        "anchor_point": anchor_point,
        "anchor_timestamp": anchor_timestamp,
        "num_marker_points": 0,
        "num_visual_corners": 4,
        "marker_pnp_input": "visual_aruco_sim3",
        "visual_marker_keyframes": int(len({int(obs["kf_id"]) for obs in observations})),
        "visual_marker_detections": int(len(observations)),
        "anchor_image_path": str(anchor_obs["image_path"]),
        "marker_pnp_rms_px": float(anchor_obs["rms_px"]),
        "visual_sim3_diagnostics": diagnostics,
    }


def fit_visual_aruco_sim3_fixed_scale(
    name: str,
    observations: List[Dict[str, object]],
    min_detections: int,
    trim_mad: float,
    fixed_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Same robust multi-keyframe rotation fit as fit_visual_aruco_sim3 (chordal-mean
    rotation average with MAD-trimmed outlier rejection), but the scale is held fixed
    at an externally-supplied value throughout every iteration instead of being fit
    jointly. Outlier trimming and the final translation are therefore self-consistent
    with the scale that will actually be used, rather than mixing an outlier set
    chosen under Sim3's own independently-fit scale with a different external one."""
    if len(observations) < min_detections:
        raise ValueError(
            f"{name}: only {len(observations)} visual ArUco detection(s), need {min_detections} "
            "for a fixed-scale visual ArUco Sim(3) rotation fit"
        )

    centers_slam = np.array([obs["center_slam"] for obs in observations], dtype=np.float64)
    centers_marker = np.array([obs["center_marker"] for obs in observations], dtype=np.float64)
    rotations = np.array([obs["R_slam_to_marker"] for obs in observations], dtype=np.float64)
    keep = np.ones(len(observations), dtype=bool)

    R_slam_to_marker = np.eye(3, dtype=np.float64)
    t_slam_to_marker = np.zeros(3, dtype=np.float64)
    residuals = np.zeros(len(observations), dtype=np.float64)

    for _ in range(5):
        if int(np.count_nonzero(keep)) < min_detections:
            break
        R_slam_to_marker = rotation_chordal_mean(rotations[keep])
        slam_rotated = centers_slam[keep] @ R_slam_to_marker.T
        marker_kept = centers_marker[keep]
        t_candidates = marker_kept - fixed_scale * slam_rotated
        t_slam_to_marker = np.median(t_candidates, axis=0)
        predicted_all = fixed_scale * (centers_slam @ R_slam_to_marker.T) + t_slam_to_marker
        residuals = np.linalg.norm(predicted_all - centers_marker, axis=1)

        if trim_mad <= 0.0:
            break
        kept_residuals = residuals[keep]
        median = float(np.median(kept_residuals))
        mad = float(np.median(np.abs(kept_residuals - median)))
        if mad <= 1e-12:
            break
        threshold = median + trim_mad * 1.4826 * mad
        new_keep = residuals <= threshold
        if int(np.count_nonzero(new_keep)) < min_detections or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    kept_residuals = residuals[keep]
    rotation_devs = np.array([rotation_angle_deg(R, R_slam_to_marker) for R in rotations], dtype=np.float64)
    diagnostics = {
        "visual_detections_total": int(len(observations)),
        "visual_detections_used": int(np.count_nonzero(keep)),
        "visual_detections_rejected": int(len(observations) - np.count_nonzero(keep)),
        "fixed_scale_m_per_slam_unit": float(fixed_scale),
        "visual_sim3_residual_m": {
            "median_all": float(np.median(residuals)),
            "mean_all": float(np.mean(residuals)),
            "max_all": float(np.max(residuals)),
            "median_used": float(np.median(kept_residuals)),
            "mean_used": float(np.mean(kept_residuals)),
            "max_used": float(np.max(kept_residuals)),
        },
        "visual_rotation_deviation_deg": {
            "median": float(np.median(rotation_devs)),
            "mean": float(np.mean(rotation_devs)),
            "max": float(np.max(rotation_devs)),
        },
    }
    print(
        f"{name}: visual ArUco Sim(3) with fixed scale = {fixed_scale:.6f} m/slam-unit "
        f"using {diagnostics['visual_detections_used']}/{diagnostics['visual_detections_total']} detection(s); "
        f"residual median={diagnostics['visual_sim3_residual_m']['median_used']:.4f} m, "
        f"rotation dev median={diagnostics['visual_rotation_deviation_deg']['median']:.3f} deg"
    )
    return R_slam_to_marker, t_slam_to_marker, keep, diagnostics


def align_camera_optimized_anchor(
    name: str,
    marker_id: int,
    raw_csv_path: Path,
    calib: CameraCalibration,
    marker_length_m: float,
    dictionary_size: int,
    dictionary_bits: int,
    min_detections: int,
    max_rms_px: float,
    max_keyframes: Optional[int],
    trim_mad: float,
    forced_scale: float,
    use_source_timestamps: bool = True,
) -> Dict[str, object]:
    """Same robust, many-keyframe rotation fit as align_camera_visual_aruco_sim3
    (marker detected across many keyframes, combined via a MAD-trimmed chordal-mean
    rotation average instead of one single anchor keyframe's PnP solve), but the
    metric scale is taken from `forced_scale` (e.g. ground-plane or point-pair scale
    recovery) instead of being fit jointly with rotation. This isolates the
    single-anchor-keyframe robustness fix from the choice of scale source: any scale
    recovery approach can be paired with this more robust rotation estimate."""
    trajectory, timestamp_source = load_trajectory(raw_csv_path, name, use_source_timestamps)
    print(f"{name}: trajectory timestamps = {timestamp_source}")
    observations = collect_visual_aruco_pose_observations(
        name,
        raw_csv_path,
        marker_id,
        calib,
        marker_length_m,
        dictionary_size,
        dictionary_bits,
        trajectory,
        max_rms_px,
        max_keyframes,
    )
    # Same robust multi-keyframe rotation fit as visual-aruco-sim3, but with the
    # scale held fixed at forced_scale throughout -- so outlier trimming and the
    # final translation are consistent with the scale we're actually using, not
    # with whatever scale Sim(3) would have fit on its own.
    R_slam_to_marker, t_slam_to_marker, keep, diagnostics = fit_visual_aruco_sim3_fixed_scale(
        name,
        observations,
        min_detections,
        trim_mad,
        forced_scale,
    )

    best_index = min(
        range(len(observations)),
        key=lambda idx: (float(observations[idx]["rms_px"]), -float(observations[idx]["area_px2"])),
    )
    anchor_obs = observations[best_index]
    kf_id = int(anchor_obs["kf_id"])
    anchor_timestamp = float(anchor_obs["timestamp"])

    items = sorted(trajectory.items(), key=lambda kv: kv[1][0])
    kf_ids_sorted = np.array([k for k, _ in items], dtype=np.int64)
    timestamps = np.array([v[0] for _, v in items], dtype=np.float64)
    centers_slam = np.array([v[1] for _, v in items], dtype=np.float64)
    rotations_slam = np.array([v[2] for _, v in items], dtype=np.float64)

    positions_marker = forced_scale * (centers_slam @ R_slam_to_marker.T) + t_slam_to_marker
    rotations_marker = rotations_slam @ R_slam_to_marker.T
    anchor_point = (
        forced_scale * (np.asarray(anchor_obs["center_slam"], dtype=np.float64) @ R_slam_to_marker.T)
        + t_slam_to_marker
    )

    return {
        "name": name,
        "marker_id": marker_id,
        "kf_id": kf_id,
        "scale": forced_scale,
        "scale_source": "external_optimized_anchor",
        "scale_recovered": True,
        "scale_warning": None,
        "R_slam_to_marker": R_slam_to_marker,
        "t_slam_to_marker": t_slam_to_marker,
        "trajectory_kf_ids": kf_ids_sorted,
        "trajectory_timestamps": timestamps,
        "trajectory_timestamp_source": timestamp_source,
        "trajectory": positions_marker,
        "trajectory_rotations": rotations_marker,
        "anchor_point": anchor_point,
        "anchor_timestamp": anchor_timestamp,
        "num_marker_points": 0,
        "num_visual_corners": 4,
        "marker_pnp_input": "optimized_anchor",
        "visual_marker_keyframes": int(len({int(obs["kf_id"]) for obs in observations})),
        "visual_marker_detections": int(len(observations)),
        "anchor_image_path": str(anchor_obs["image_path"]),
        "marker_pnp_rms_px": float(anchor_obs["rms_px"]),
        "visual_sim3_diagnostics": diagnostics,
    }


def align_camera_visual_aruco_motion_scale(
    name: str,
    marker_id: int,
    raw_csv_path: Path,
    calib: CameraCalibration,
    marker_length_m: float,
    dictionary_size: int,
    dictionary_bits: int,
    min_detections: int,
    max_rms_px: float,
    max_keyframes: Optional[int],
    trim_mad: float,
    motion_min_distance_m: float,
    motion_max_pairs: int,
    use_source_timestamps: bool = True,
) -> Dict[str, object]:
    trajectory, timestamp_source = load_trajectory(raw_csv_path, name, use_source_timestamps)
    print(f"{name}: trajectory timestamps = {timestamp_source}")
    observations = collect_visual_aruco_pose_observations(
        name,
        raw_csv_path,
        marker_id,
        calib,
        marker_length_m,
        dictionary_size,
        dictionary_bits,
        trajectory,
        max_rms_px,
        max_keyframes,
    )
    scale, R_slam_to_marker, t_slam_to_marker, keep, diagnostics = fit_visual_aruco_motion_scale(
        name,
        observations,
        min_detections,
        trim_mad,
        motion_min_distance_m,
        motion_max_pairs,
    )

    best_index = min(range(len(observations)), key=lambda idx: (float(observations[idx]["rms_px"]), -float(observations[idx]["area_px2"])))
    anchor_obs = observations[best_index]
    kf_id = int(anchor_obs["kf_id"])
    anchor_timestamp = float(anchor_obs["timestamp"])

    items = sorted(trajectory.items(), key=lambda kv: kv[1][0])
    kf_ids_sorted = np.array([k for k, _ in items], dtype=np.int64)
    timestamps = np.array([v[0] for _, v in items], dtype=np.float64)
    centers_slam = np.array([v[1] for _, v in items], dtype=np.float64)
    rotations_slam = np.array([v[2] for _, v in items], dtype=np.float64)

    positions_marker = scale * (centers_slam @ R_slam_to_marker.T) + t_slam_to_marker
    rotations_marker = rotations_slam @ R_slam_to_marker.T
    anchor_point = scale * (np.asarray(anchor_obs["center_slam"], dtype=np.float64) @ R_slam_to_marker.T) + t_slam_to_marker

    return {
        "name": name,
        "marker_id": marker_id,
        "kf_id": kf_id,
        "scale": scale,
        "scale_source": "visual_aruco_motion_scale",
        "scale_recovered": True,
        "scale_warning": None,
        "R_slam_to_marker": R_slam_to_marker,
        "t_slam_to_marker": t_slam_to_marker,
        "trajectory_kf_ids": kf_ids_sorted,
        "trajectory_timestamps": timestamps,
        "trajectory_timestamp_source": timestamp_source,
        "trajectory": positions_marker,
        "trajectory_rotations": rotations_marker,
        "anchor_point": anchor_point,
        "anchor_timestamp": anchor_timestamp,
        "num_marker_points": 0,
        "num_visual_corners": 4,
        "marker_pnp_input": "visual_aruco_motion_scale",
        "visual_marker_keyframes": int(len({int(obs["kf_id"]) for obs in observations})),
        "visual_marker_detections": int(len(observations)),
        "anchor_image_path": str(anchor_obs["image_path"]),
        "marker_pnp_rms_px": float(anchor_obs["rms_px"]),
        "visual_motion_scale_diagnostics": diagnostics,
    }


def plot_anchor_distance_cutoff(ax, center: np.ndarray, radius: float, color: str, label: Optional[str] = None) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    pts = np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full_like(theta, center[2]),
        ]
    )
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color=color,
        linestyle="--",
        linewidth=0.9,
        alpha=0.55,
        label=label,
    )


def plot_anchor_distance_cutoff_2d(ax, center: np.ndarray, radius: float, color: str, label: Optional[str] = None) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color=color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label=label,
    )


def trajectory_cutoff_mask(
    traj: np.ndarray,
    timestamps: Optional[np.ndarray],
    anchor: np.ndarray,
    anchor_timestamp: Optional[float],
    max_distance_from_anchor_m: Optional[float],
    max_time_from_anchor_s: Optional[float],
) -> Optional[np.ndarray]:
    masks: List[np.ndarray] = []
    if max_distance_from_anchor_m is not None:
        distances = np.linalg.norm(traj - anchor.reshape(1, 3), axis=1)
        masks.append(distances <= max_distance_from_anchor_m)
    if max_time_from_anchor_s is not None and timestamps is not None and anchor_timestamp is not None:
        masks.append(np.abs(timestamps - anchor_timestamp) <= max_time_from_anchor_s)
    if not masks:
        return None
    active = masks[0].copy()
    for mask in masks[1:]:
        active &= mask
    return active


def cutoff_label(max_distance_from_anchor_m: Optional[float], max_time_from_anchor_s: Optional[float]) -> str:
    parts = []
    if max_distance_from_anchor_m is not None:
        parts.append(f"{max_distance_from_anchor_m:g}m")
    if max_time_from_anchor_s is not None:
        parts.append(f"{max_time_from_anchor_s:g}s")
    return " + ".join(parts) if parts else "cutoff"


def camera_distance_cutoff(args: argparse.Namespace, camera_index: int) -> Optional[float]:
    if camera_index == 1 and args.camera1_max_distance_from_anchor_m is not None:
        return args.camera1_max_distance_from_anchor_m
    if camera_index == 2 and args.camera2_max_distance_from_anchor_m is not None:
        return args.camera2_max_distance_from_anchor_m
    return args.max_distance_from_anchor_m


def camera_time_cutoff(args: argparse.Namespace, camera_index: int) -> Optional[float]:
    if camera_index == 1 and args.camera1_max_time_from_anchor_s is not None:
        return args.camera1_max_time_from_anchor_s
    if camera_index == 2 and args.camera2_max_time_from_anchor_s is not None:
        return args.camera2_max_time_from_anchor_s
    return args.max_time_from_anchor_s


def plot_trajectory_line_2d(
    ax,
    traj: np.ndarray,
    color: str,
    label: str,
    timestamps: Optional[np.ndarray],
    anchor: np.ndarray,
    anchor_timestamp: Optional[float],
    max_distance_from_anchor_m: Optional[float],
    max_time_from_anchor_s: Optional[float],
    surviving_match_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    active = trajectory_cutoff_mask(
        traj,
        timestamps,
        anchor,
        anchor_timestamp,
        max_distance_from_anchor_m,
        max_time_from_anchor_s,
    )
    if active is None:
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.4, label=label)
        return traj

    cutoff_text = cutoff_label(max_distance_from_anchor_m, max_time_from_anchor_s)
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.7, alpha=0.16, label=f"{label} outside {cutoff_text}")
    if surviving_match_points is not None and len(surviving_match_points):
        points = np.asarray(surviving_match_points, dtype=np.float64)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.7,
            marker="o",
            markersize=3.2,
            label=f"{label} surviving matched timestamps",
            zorder=8,
        )
        return points

    active_traj = traj.copy()
    active_traj[~active] = np.nan
    ax.plot(active_traj[:, 0], active_traj[:, 1], color=color, linewidth=2.0, label=f"{label} within {cutoff_text}")
    return traj[active]


def plot_trajectories(
    results: List[Dict[str, object]],
    marker_id: int,
    marker_length_m: float,
    output_plot: Path,
    show: bool,
    max_distance_from_anchor_m: Optional[float] = None,
    per_camera_max_distance_from_anchor_m: Optional[List[Optional[float]]] = None,
    max_time_from_anchor_s: Optional[float] = None,
    per_camera_max_time_from_anchor_s: Optional[List[Optional[float]]] = None,
    surviving_match_points: Optional[List[np.ndarray]] = None,
    show_anchor_marker: bool = True,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    half = marker_length_m * 0.5
    square = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0], [-half, half, 0.0]]
    )
    ax.plot(square[:, 0], square[:, 1], square[:, 2], color="black", linewidth=2, label=f"marker {marker_id}")
    ax.scatter(
        0.0,
        0.0,
        0.0,
        color="black",
        marker="s",
        s=80,
        label="marker origin (0,0,0)",
        zorder=10,
    )

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    all_points = [square]
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        traj = np.asarray(res["trajectory"], dtype=np.float64)
        anchor = np.asarray(res["anchor_point"], dtype=np.float64)
        timestamps = np.asarray(res["trajectory_timestamps"], dtype=np.float64)
        anchor_timestamp = float(res["anchor_timestamp"])
        distance_cutoff = (
            per_camera_max_distance_from_anchor_m[i]
            if per_camera_max_distance_from_anchor_m is not None
            else max_distance_from_anchor_m
        )
        time_cutoff = (
            per_camera_max_time_from_anchor_s[i]
            if per_camera_max_time_from_anchor_s is not None
            else max_time_from_anchor_s
        )
        cutoff_text = cutoff_label(distance_cutoff, time_cutoff)
        active = trajectory_cutoff_mask(
            traj,
            timestamps,
            anchor,
            anchor_timestamp,
            distance_cutoff,
            time_cutoff,
        )
        if active is None:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1.5, label=f"{res['name']} trajectory")
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=color,
                linewidth=1.5,
                label=f"{res['name']} trajectory",
            )
            active_traj = traj.copy()
            active_traj[~active] = np.nan
            ax.plot(
                active_traj[:, 0],
                active_traj[:, 1],
                active_traj[:, 2],
                color=color,
                linewidth=2.8,
                label=f"{res['name']} within {cutoff_text}",
            )
        if show_anchor_marker:
            ax.scatter(anchor[0], anchor[1], anchor[2], color=color, marker="*", s=220, edgecolor="black",
                        label=f"{res['name']} kf{res['kf_id']} anchor")
        if distance_cutoff is not None:
            plot_anchor_distance_cutoff(
                ax,
                np.asarray(anchor, dtype=np.float64),
                float(distance_cutoff),
                color,
                label=f"{res['name']} {distance_cutoff:g}m anchor cutoff",
            )
        all_points.append(traj)
        if distance_cutoff is not None:
            all_points.append(np.asarray([anchor], dtype=np.float64) + np.array(
                [
                    [distance_cutoff, 0.0, 0.0],
                    [-distance_cutoff, 0.0, 0.0],
                    [0.0, distance_cutoff, 0.0],
                    [0.0, -distance_cutoff, 0.0],
                    [0.0, 0.0, distance_cutoff],
                    [0.0, 0.0, -distance_cutoff],
                ],
                dtype=np.float64,
            ))

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    title_parts = [f"Camera trajectories in ArUco marker {marker_id} frame"]
    if per_camera_max_distance_from_anchor_m is not None:
        title_parts.append(
            "distance cutoffs: "
            + ", ".join(
                f"{results[i]['name']}={cutoff:g} m"
                for i, cutoff in enumerate(per_camera_max_distance_from_anchor_m)
                if cutoff is not None
            )
        )
    elif max_distance_from_anchor_m is not None:
        title_parts.append(f"distance cutoff: {max_distance_from_anchor_m:g} m")
    if per_camera_max_time_from_anchor_s is not None:
        title_parts.append(
            "time cutoffs: "
            + ", ".join(
                f"{results[i]['name']}={cutoff:g} s"
                for i, cutoff in enumerate(per_camera_max_time_from_anchor_s)
                if cutoff is not None
            )
        )
    elif max_time_from_anchor_s is not None:
        title_parts.append(f"time cutoff: {max_time_from_anchor_s:g} s")
    ax.set_title("\n".join(title_parts))
    ax.legend(loc="upper left", fontsize=8)

    stacked = np.vstack(all_points)
    center = stacked.mean(axis=0)
    radius = float(np.max(np.linalg.norm(stacked - center, axis=1))) or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_plot}")

    topdown_plot = output_plot.with_name(f"{output_plot.stem}_topdown{output_plot.suffix}")
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    ax2.plot(square[:, 0], square[:, 1], color="black", linewidth=2, label=f"marker {marker_id}")
    ax2.scatter(0.0, 0.0, color="black", marker="s", s=70, label="marker origin (0,0)")

    zoom_points = [square[:, :2], np.array([[0.0, 0.0]], dtype=np.float64)]
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        traj = np.asarray(res["trajectory"], dtype=np.float64)
        timestamps = np.asarray(res["trajectory_timestamps"], dtype=np.float64)
        anchor = np.asarray(res["anchor_point"], dtype=np.float64)
        anchor_timestamp = float(res["anchor_timestamp"])
        distance_cutoff = (
            per_camera_max_distance_from_anchor_m[i]
            if per_camera_max_distance_from_anchor_m is not None
            else max_distance_from_anchor_m
        )
        time_cutoff = (
            per_camera_max_time_from_anchor_s[i]
            if per_camera_max_time_from_anchor_s is not None
            else max_time_from_anchor_s
        )

        near_time_points = plot_trajectory_line_2d(
            ax2,
            traj,
            color,
            f"{res['name']} trajectory",
            timestamps,
            anchor,
            anchor_timestamp,
            distance_cutoff,
            time_cutoff,
            surviving_match_points[i] if surviving_match_points is not None else None,
        )
        if show_anchor_marker:
            ax2.scatter(
                anchor[0],
                anchor[1],
                color=color,
                marker="*",
                s=170,
                edgecolor="black",
                label=f"{res['name']} kf{res['kf_id']} anchor",
                zorder=10,
            )
        if distance_cutoff is not None:
            plot_anchor_distance_cutoff_2d(
                ax2,
                anchor,
                float(distance_cutoff),
                color,
                label=f"{res['name']} {distance_cutoff:g}m cutoff",
            )
            zoom_points.append(
                np.array(
                    [
                        [anchor[0] + distance_cutoff, anchor[1]],
                        [anchor[0] - distance_cutoff, anchor[1]],
                        [anchor[0], anchor[1] + distance_cutoff],
                        [anchor[0], anchor[1] - distance_cutoff],
                    ],
                    dtype=np.float64,
                )
            )
        else:
            distance_to_anchor = np.linalg.norm(traj[:, :2] - anchor[:2].reshape(1, 2), axis=1)
            if len(distance_to_anchor):
                local_radius = max(float(np.percentile(distance_to_anchor, 20)), 1.0)
                local_mask = distance_to_anchor <= local_radius
                if np.any(local_mask):
                    zoom_points.append(traj[local_mask, :2])
        if len(near_time_points):
            zoom_points.append(near_time_points[:, :2])
        zoom_points.append(anchor[:2].reshape(1, 2))

    title_parts_2d = [f"Top-down trajectories in ArUco marker {marker_id} frame"]
    if per_camera_max_distance_from_anchor_m is not None:
        title_parts_2d.append(
            "distance cutoffs: "
            + ", ".join(
                f"{results[i]['name']}={cutoff:g} m"
                for i, cutoff in enumerate(per_camera_max_distance_from_anchor_m)
                if cutoff is not None
            )
        )
    elif max_distance_from_anchor_m is not None:
        title_parts_2d.append(f"distance cutoff: {max_distance_from_anchor_m:g} m")
    if per_camera_max_time_from_anchor_s is not None:
        title_parts_2d.append(
            "time cutoffs: "
            + ", ".join(
                f"{results[i]['name']}={cutoff:g} s"
                for i, cutoff in enumerate(per_camera_max_time_from_anchor_s)
                if cutoff is not None
            )
        )
    elif max_time_from_anchor_s is not None:
        title_parts_2d.append(f"time cutoff: {max_time_from_anchor_s:g} s")
    ax2.set_title("\n".join(title_parts_2d))
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linewidth=0.4, alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8)

    zoom_stacked = np.vstack([pts for pts in zoom_points if len(pts)])
    xy_min = zoom_stacked.min(axis=0)
    xy_max = zoom_stacked.max(axis=0)
    span = xy_max - xy_min
    max_span = float(max(span.max(), marker_length_m * 2.0, 1.0))
    margin = max_span * 0.15
    center_xy = (xy_min + xy_max) * 0.5
    half_span = max_span * 0.5 + margin
    ax2.set_xlim(center_xy[0] - half_span, center_xy[0] + half_span)
    ax2.set_ylim(center_xy[1] - half_span, center_xy[1] + half_span)

    fig2.savefig(topdown_plot, dpi=180, bbox_inches="tight")
    print(f"Saved top-down plot: {topdown_plot}")
    if show:
        plt.show()


# --------------------------------------------------------------------------------------
# Stage 2: extrinsic calibration (reuses stage 1's per-camera alignment results)
# --------------------------------------------------------------------------------------


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_pose(
    timestamps: np.ndarray, positions: np.ndarray, rotations: np.ndarray, query_ts: float
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Returns (position, rotation, bracket_width_s) interpolated at query_ts, or
    None if query_ts falls outside [timestamps[0], timestamps[-1]]."""
    if query_ts < timestamps[0] or query_ts > timestamps[-1]:
        return None
    i = int(np.searchsorted(timestamps, query_ts))
    if i == 0:
        i = 1
    t0, t1 = timestamps[i - 1], timestamps[i]
    alpha = 0.0 if t1 == t0 else (query_ts - t0) / (t1 - t0)
    pos = positions[i - 1] * (1 - alpha) + positions[i] * alpha
    q0 = matrix_to_quaternion(rotations[i - 1])
    q1 = matrix_to_quaternion(rotations[i])
    R = quaternion_to_matrix(slerp(q0, q1, alpha))
    return pos, R, float(t1 - t0)


def relative_extrinsic(R_a: np.ndarray, C_a: np.ndarray, R_b: np.ndarray, C_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """p_a_cam = R_b_to_a @ p_b_cam + t_b_to_a, given two world-to-camera rotations
    (R_a, R_b) and CAMERA CENTERS (C_a, C_b, i.e. positions, not the tcw translation
    vector) expressed in the same world (marker) frame.

    p_x_cam = R_x @ (p_world - C_x), so substituting p_world = R_b.T @ p_b_cam + C_b
    into camera a's equation gives R_b_to_a = R_a @ R_b.T and
    t_b_to_a = R_a @ (C_b - C_a) -- NOT "C_a - R_b_to_a @ C_b", which would be correct
    only if C_a/C_b were already the tcw vectors (t = -R @ C) rather than centers.
    """
    R_b_to_a = R_a @ R_b.T
    t_b_to_a = R_a @ (C_b - C_a)
    return R_b_to_a, t_b_to_a


def rotation_chordal_mean(rotations: np.ndarray) -> np.ndarray:
    M = rotations.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_diff = R_a @ R_b.T
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = min(max(cos_angle, -1.0), 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def summarize(rotations: np.ndarray, translations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_mean = rotation_chordal_mean(rotations)
    t_median = np.median(translations, axis=0)
    angle_devs = np.array([rotation_angle_deg(R, R_mean) for R in rotations])
    trans_devs = np.linalg.norm(translations - t_median, axis=1)
    return R_mean, t_median, angle_devs, trans_devs


def optimize_extrinsic(
    rotations: np.ndarray,
    translations: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    loss: str = "huber",
    f_scale_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Stage 3: joint robust refinement over SE(3). Each match's relative_extrinsic()
    result (rotations[i], translations[i]) is already a complete, closed-form estimate of
    the same camera2->camera1 extrinsic -- there are no free unknowns entangled in raw
    poses left to solve jointly (unlike e.g. optim.txt's multi-tag reprojection problem,
    where the offsets genuinely couple through a nonlinear reprojection function).
    summarize() above instead combines the N noisy per-match estimates with two
    independent, mismatched rules: an SVD chordal mean for rotation and a per-axis median
    for translation. This replaces that with a single (R, t) minimizing one combined
    robust (default Huber) loss over rotation and translation residuals *together*, per
    match -- a joint M-estimator on SE(3) instead of two separately-reduced statistics,
    smoothly downweighting outlier matches instead of relying only on the hard
    anchor-distance/time cutoffs upstream. Parameterized as rvec (axis-angle,
    unconstrained) + t, same pattern as optim.txt's packT/unpackT via
    rotm2axang/axang2rotm.
    """
    n = len(rotations)

    # Rotation residuals are naturally in radians and translation residuals in meters --
    # not comparable units. Scale rotation residuals by the ratio of their typical (median)
    # magnitude to translation's, at the initial guess, so neither term dominates the
    # objective purely because of unit choice rather than actual fit quality.
    rot_res0_deg = np.array([rotation_angle_deg(R_init, rotations[i]) for i in range(n)])
    trans_res0 = np.linalg.norm(translations - t_init, axis=1)
    rot_scale = float(np.median(rot_res0_deg)) * np.pi / 180.0
    trans_scale = float(np.median(trans_res0))
    rot_weight = trans_scale / rot_scale if rot_scale > 1e-9 else 1.0

    if f_scale_m is None:
        f_scale_m = max(1.5 * trans_scale, 1e-4)

    def residuals(x: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(x[:3])
        t = x[3:]
        res = np.empty((n, 6))
        for i in range(n):
            rot_err_vec, _ = cv2.Rodrigues(R.T @ rotations[i])
            res[i, :3] = rot_weight * rot_err_vec.flatten()
            res[i, 3:] = t - translations[i]
        return res.flatten()

    rvec_init, _ = cv2.Rodrigues(R_init)
    x0 = np.concatenate([rvec_init.flatten(), t_init])

    initial_cost = float(0.5 * np.sum(residuals(x0) ** 2))
    result = least_squares(residuals, x0, loss=loss, f_scale=f_scale_m, method="trf")

    R_opt, _ = cv2.Rodrigues(result.x[:3])
    t_opt = result.x[3:]

    diagnostics = {
        "loss": loss,
        "f_scale_m": float(f_scale_m),
        "rot_weight": float(rot_weight),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "initial_cost": initial_cost,
        "final_cost": float(result.cost),
        "rotation_change_from_average_deg": rotation_angle_deg(R_opt, R_init),
        "translation_change_from_average_m": float(np.linalg.norm(t_opt - t_init)),
    }
    return R_opt, t_opt, diagnostics


def _robust_rho(loss: str, z: np.ndarray) -> np.ndarray:
    """Same rho(z) family scipy.optimize.least_squares uses internally (z = (r/f_scale)^2),
    exposed here so optimize_extrinsic_grouped() can apply it to a *combined* per-match
    residual norm instead of per-component, while staying directly comparable (same loss
    shape, same f_scale) to optimize_extrinsic()'s result."""
    if loss == "linear":
        return z
    if loss == "huber":
        return np.where(z <= 1, z, 2 * np.sqrt(z) - 1)
    if loss == "soft_l1":
        return 2 * (np.sqrt(1 + z) - 1)
    if loss == "cauchy":
        return np.log1p(z)
    if loss == "arctan":
        return np.arctan(z)
    raise ValueError(f"Unknown loss: {loss}")


def optimize_extrinsic_grouped(
    rotations: np.ndarray,
    translations: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    loss: str = "huber",
    f_scale_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Stage 3b: joint SE(3) refinement with a *grouped* per-match robust loss, solved
    with a generic scalar minimizer (scipy.optimize.minimize, BFGS -- the Python analog
    of MATLAB's fminunc) instead of scipy.optimize.least_squares.

    optimize_extrinsic() above applies the robust loss to each of the 6 residual
    *components* (3 rotation + 3 translation) independently, because that is what
    least_squares's API supports -- a match could have one component judged an inlier
    and another an outlier. This instead computes ONE combined 6D residual norm per
    match and applies the robust loss to that single number, so a match is downweighted
    or effectively rejected as a whole -- exactly the thing a generic minimizer can do
    that least_squares's per-component loss structurally cannot. Same rho(z) family and
    f_scale as optimize_extrinsic() so the two are directly comparable.
    """
    n = len(rotations)
    rot_res0_deg = np.array([rotation_angle_deg(R_init, rotations[i]) for i in range(n)])
    trans_res0 = np.linalg.norm(translations - t_init, axis=1)
    rot_scale = float(np.median(rot_res0_deg)) * np.pi / 180.0
    trans_scale = float(np.median(trans_res0))
    rot_weight = trans_scale / rot_scale if rot_scale > 1e-9 else 1.0

    if f_scale_m is None:
        f_scale_m = max(1.5 * trans_scale, 1e-4)

    def match_residual_norms(x: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(x[:3])
        t = x[3:]
        norms = np.empty(n)
        for i in range(n):
            rot_err_vec, _ = cv2.Rodrigues(R.T @ rotations[i])
            res6 = np.concatenate([rot_weight * rot_err_vec.flatten(), t - translations[i]])
            norms[i] = np.linalg.norm(res6)
        return norms

    def cost(x: np.ndarray) -> float:
        norms = match_residual_norms(x)
        z = (norms / f_scale_m) ** 2
        return float(0.5 * f_scale_m**2 * np.sum(_robust_rho(loss, z)))

    rvec_init, _ = cv2.Rodrigues(R_init)
    x0 = np.concatenate([rvec_init.flatten(), t_init])
    initial_cost = cost(x0)

    # L-BFGS-B (unconstrained here, no bounds given) rather than plain BFGS: with a
    # numerically-differentiated gradient (cost() has no analytical jac), BFGS's line
    # search reports "precision loss" / success=False right at the true optimum once the
    # finite-difference gradient noise floor is reached, even though the solution is
    # correct (verified: same cost/solution as BFGS, but with a clean convergence flag).
    result = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-12})

    R_opt, _ = cv2.Rodrigues(result.x[:3])
    t_opt = result.x[3:]

    diagnostics = {
        "loss": loss,
        "f_scale_m": float(f_scale_m),
        "rot_weight": float(rot_weight),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "initial_cost": initial_cost,
        "final_cost": float(result.fun),
        "rotation_change_from_average_deg": rotation_angle_deg(R_opt, R_init),
        "translation_change_from_average_m": float(np.linalg.norm(t_opt - t_init)),
    }
    return R_opt, t_opt, diagnostics


def rotation_matrix_to_euler_zyx_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- the roll/pitch/yaw convention
    used by Agilex Recordings/Intrinsic_ground_truths/robot_relative_extrinsics.yaml
    (verified against its T_front_left entry to float precision)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def format_extrinsic_yaml(camera1: str, camera2: str, R: np.ndarray, t: np.ndarray, suffix: str = "") -> str:
    """Formats camera2->camera1 as T_<camera1>_<camera2>, matching
    robot_relative_extrinsics.yaml's layout exactly (from_frame/to_frame/
    translation_xyz/euler_zyx_deg/matrix). suffix (e.g. "_optimized") is appended to the
    block's key only, so the stage-2 (averaged) and stage-3 (optimized) results can be
    saved side by side in the same file without colliding."""
    roll, pitch, yaw = rotation_matrix_to_euler_zyx_deg(R)

    def f(x: float) -> str:
        return f"{x:.12f}"

    lines = [
        f"T_{camera1}_{camera2}{suffix}:",
        f"    from_frame: {camera2}",
        f"    to_frame: {camera1}",
        f"    translation_xyz: [{f(t[0])}, {f(t[1])}, {f(t[2])}]",
        "    euler_zyx_deg:",
        f"      roll: {f(roll)}",
        f"      pitch: {f(pitch)}",
        f"      yaw: {f(yaw)}",
        "    matrix:",
    ]
    for i in range(3):
        lines.append(f"      - [{f(R[i, 0])}, {f(R[i, 1])}, {f(R[i, 2])}, {f(t[i])}]")
    lines.append(f"      - [{f(0.0)}, {f(0.0)}, {f(0.0)}, {f(1.0)}]")
    return "\n".join(lines) + "\n"


def compute_extrinsic(
    args: argparse.Namespace, marker_id: int, result1: Dict[str, object], result2: Dict[str, object]
) -> Tuple[str, Dict[str, object], Dict[str, np.ndarray]]:
    ts1, pos1, rot1 = result1["trajectory_timestamps"], result1["trajectory"], result1["trajectory_rotations"]
    ts2, pos2, rot2 = result2["trajectory_timestamps"], result2["trajectory"], result2["trajectory_rotations"]
    kf_ids2 = result2["trajectory_kf_ids"]
    anchor1, anchor2 = result1["anchor_point"], result2["anchor_point"]
    anchor1_ts, anchor2_ts = result1["anchor_timestamp"], result2["anchor_timestamp"]

    # First pass: collect every candidate match (only gated by --max-bracket-width-s, a cheap
    # and independent quality gate), keeping distance-from-anchor and time-from-anchor as
    # metadata rather than filtering by them yet -- otherwise the sensitivity sweeps below would
    # be comparing a cutoff against a set that's already been cut to a tighter one.
    matches: List[Dict[str, object]] = []
    skipped_out_of_range = 0
    skipped_bracket = 0
    for idx2 in range(len(ts2)):
        t = ts2[idx2]
        interp = interpolate_pose(ts1, pos1, rot1, t)
        if interp is None:
            skipped_out_of_range += 1
            continue
        pos1_interp, rot1_interp, bracket_width = interp
        if args.max_bracket_width_s is not None and bracket_width > args.max_bracket_width_s:
            skipped_bracket += 1
            continue
        camera1_distance_from_anchor = float(np.linalg.norm(pos1_interp - anchor1))
        camera2_distance_from_anchor = float(np.linalg.norm(pos2[idx2] - anchor2))
        distance_from_anchor = max(camera1_distance_from_anchor, camera2_distance_from_anchor)
        # Euclidean distance is only a proxy for accumulated SLAM drift -- drift tracks
        # trajectory/time distance from the anchor, not physical proximity, so a keyframe can
        # revisit the anchor's location much later (Euclidean-close) while carrying far more
        # drift than the distance filter alone would catch. Time-from-anchor is independent.
        camera1_time_from_anchor = abs(t - anchor1_ts)
        camera2_time_from_anchor = abs(float(ts2[idx2]) - anchor2_ts)
        time_from_anchor = max(camera1_time_from_anchor, camera2_time_from_anchor)
        R_rel, t_rel = relative_extrinsic(rot1_interp, pos1_interp, rot2[idx2], pos2[idx2])
        matches.append(
            {
                "kf2_id": int(kf_ids2[idx2]),
                "timestamp": float(t),
                "bracket_width_s": bracket_width,
                "camera1_position": pos1_interp,
                "camera2_position": pos2[idx2],
                "distance_from_anchor_m": distance_from_anchor,
                "camera1_distance_from_anchor_m": camera1_distance_from_anchor,
                "camera2_distance_from_anchor_m": camera2_distance_from_anchor,
                "time_from_anchor_s": time_from_anchor,
                "camera1_time_from_anchor_s": camera1_time_from_anchor,
                "camera2_time_from_anchor_s": camera2_time_from_anchor,
                "R": R_rel,
                "t": t_rel,
            }
        )

    print(
        f"\n{len(ts2)} {args.camera2_name} keyframes; {skipped_out_of_range} outside {args.camera1_name}'s "
        f"time span; {skipped_bracket} dropped by --max-bracket-width-s; {len(matches)} candidate matches"
    )
    if len(matches) < 3:
        raise SystemExit("Not enough matches to compute a robust extrinsic estimate")

    rotations = np.array([m["R"] for m in matches])
    translations = np.array([m["t"] for m in matches])
    distances = np.array([m["distance_from_anchor_m"] for m in matches])
    camera1_distances = np.array([m["camera1_distance_from_anchor_m"] for m in matches])
    camera2_distances = np.array([m["camera2_distance_from_anchor_m"] for m in matches])
    time_distances = np.array([m["time_from_anchor_s"] for m in matches])
    camera1_time_distances = np.array([m["camera1_time_from_anchor_s"] for m in matches])
    camera2_time_distances = np.array([m["camera2_time_from_anchor_s"] for m in matches])

    print(
        "\nDistance from anchor keyframe is one quality signal (monocular SLAM scale/pose drift "
        "grows with distance from the marker anchor). Sensitivity to that cutoff, over all "
        f"{len(matches)} candidate matches:"
    )
    print(f"  {'cutoff(m)':>10}  {'n':>5}  {'|t| median(m)':>15}  {'rot_dev_median(deg)':>20}  {'trans_dev_median(mm)':>20}")
    for cutoff in sorted({0.5, 1.0, 2.0, 5.0, float(distances.max())}):
        mask = distances <= cutoff
        if mask.sum() < 3:
            continue
        _, t_med, a, d = summarize(rotations[mask], translations[mask])
        print(f"  {cutoff:10.2f}  {mask.sum():5d}  {np.linalg.norm(t_med):15.3f}  {np.median(a):20.3f}  {np.median(d) * 1000:20.1f}")

    print(
        "\nTime from anchor keyframe is an independent quality signal (drift accumulates with "
        "trajectory distance, which Euclidean distance alone can miss -- e.g. a later revisit of "
        f"the anchor's location). Sensitivity to that cutoff, over all {len(matches)} candidate matches:"
    )
    print(f"  {'cutoff(s)':>10}  {'n':>5}  {'|t| median(m)':>15}  {'rot_dev_median(deg)':>20}  {'trans_dev_median(mm)':>20}")
    for cutoff in sorted({1.0, 5.0, 15.0, 30.0, float(time_distances.max())}):
        mask = time_distances <= cutoff
        if mask.sum() < 3:
            continue
        _, t_med, a, d = summarize(rotations[mask], translations[mask])
        print(f"  {cutoff:10.2f}  {mask.sum():5d}  {np.linalg.norm(t_med):15.3f}  {np.median(a):20.3f}  {np.median(d) * 1000:20.1f}")

    # Second pass: apply the user's chosen cutoffs (if any) to select the matches that actually
    # determine the reported/saved extrinsic. Both filters are independent and combined with AND.
    final_mask = np.ones(len(matches), dtype=bool)
    camera1_distance_cutoff = camera_distance_cutoff(args, 1)
    camera2_distance_cutoff = camera_distance_cutoff(args, 2)
    camera1_time_cutoff = camera_time_cutoff(args, 1)
    camera2_time_cutoff = camera_time_cutoff(args, 2)
    if camera1_distance_cutoff is not None:
        final_mask &= camera1_distances <= camera1_distance_cutoff
    if camera2_distance_cutoff is not None:
        final_mask &= camera2_distances <= camera2_distance_cutoff
    if camera1_time_cutoff is not None:
        final_mask &= camera1_time_distances <= camera1_time_cutoff
    if camera2_time_cutoff is not None:
        final_mask &= camera2_time_distances <= camera2_time_cutoff
    skipped_distance = int((~final_mask).sum())
    (
        rotations,
        translations,
        distances,
        camera1_distances,
        camera2_distances,
        time_distances,
        camera1_time_distances,
        camera2_time_distances,
    ) = (
        rotations[final_mask],
        translations[final_mask],
        distances[final_mask],
        camera1_distances[final_mask],
        camera2_distances[final_mask],
        time_distances[final_mask],
        camera1_time_distances[final_mask],
        camera2_time_distances[final_mask],
    )
    matches = [m for m, keep in zip(matches, final_mask) if keep]
    print(
        f"\n{skipped_distance} candidate matches dropped by anchor distance/time cutoffs; "
        f"{len(matches)} used for the final extrinsic"
    )
    if camera1_distance_cutoff is not None or camera2_distance_cutoff is not None:
        print(
            "Distance cutoffs used: "
            f"{args.camera1_name}={camera1_distance_cutoff if camera1_distance_cutoff is not None else 'none'} m, "
            f"{args.camera2_name}={camera2_distance_cutoff if camera2_distance_cutoff is not None else 'none'} m"
        )
    if camera1_time_cutoff is not None or camera2_time_cutoff is not None:
        print(
            "Time cutoffs used: "
            f"{args.camera1_name}={camera1_time_cutoff if camera1_time_cutoff is not None else 'none'} s, "
            f"{args.camera2_name}={camera2_time_cutoff if camera2_time_cutoff is not None else 'none'} s"
        )
    if len(matches) < 3:
        raise SystemExit("Not enough matches survive the anchor-distance/time cutoffs for a robust estimate")

    R_mean, t_median, angle_devs, trans_devs = summarize(rotations, translations)
    print(
        f"\nUsing all {len(matches)} matches: rotation deviation from mean "
        f"(deg) median={np.median(angle_devs):.3f} mean={angle_devs.mean():.3f} max={angle_devs.max():.3f}"
    )
    print(
        f"translation deviation from median (m) median={np.median(trans_devs):.4f} "
        f"mean={trans_devs.mean():.4f} max={trans_devs.max():.4f}"
    )

    R_opt, t_opt, opt_diag = optimize_extrinsic(
        rotations, translations, R_mean, t_median, loss=args.optimize_loss, f_scale_m=args.optimize_f_scale_m
    )
    print(
        f"\nStage 3 joint optimization ({opt_diag['loss']} loss, f_scale={opt_diag['f_scale_m'] * 1000:.1f}mm): "
        f"cost {opt_diag['initial_cost']:.6f} -> {opt_diag['final_cost']:.6f} "
        f"({opt_diag['nfev']} evals, success={opt_diag['success']}); moved "
        f"{opt_diag['rotation_change_from_average_deg']:.3f} deg / "
        f"{opt_diag['translation_change_from_average_m'] * 1000:.1f} mm from the averaged (stage 2) result"
    )

    R_grp, t_grp, grp_diag = optimize_extrinsic_grouped(
        rotations, translations, R_mean, t_median, loss=args.optimize_loss, f_scale_m=args.optimize_f_scale_m
    )
    print(
        f"\nStage 3b grouped joint optimization ({grp_diag['loss']} loss, f_scale={grp_diag['f_scale_m'] * 1000:.1f}mm): "
        f"cost {grp_diag['initial_cost']:.6f} -> {grp_diag['final_cost']:.6f} "
        f"({grp_diag['nfev']} evals, success={grp_diag['success']}); moved "
        f"{grp_diag['rotation_change_from_average_deg']:.3f} deg / "
        f"{grp_diag['translation_change_from_average_m'] * 1000:.1f} mm from the averaged (stage 2) result"
    )

    extrinsic_yaml_averaged = format_extrinsic_yaml(args.camera1_name, args.camera2_name, R_mean, t_median)
    extrinsic_yaml_optimized = format_extrinsic_yaml(
        args.camera1_name, args.camera2_name, R_opt, t_opt, suffix="_optimized"
    )
    extrinsic_yaml_grouped = format_extrinsic_yaml(
        args.camera1_name, args.camera2_name, R_grp, t_grp, suffix="_optimized_grouped"
    )
    print(f"\n{extrinsic_yaml_averaged}")
    print(f"\n{extrinsic_yaml_optimized}")
    print(f"\n{extrinsic_yaml_grouped}")
    extrinsic_yaml = extrinsic_yaml_averaged + "\n" + extrinsic_yaml_optimized + "\n" + extrinsic_yaml_grouped

    diagnostics = {
        "marker_id": marker_id,
        "camera1": args.camera1_name,
        "camera2": args.camera2_name,
        "alignment_source": args.alignment_source,
        "scale_ambiguous": bool((not result1["scale_recovered"]) or (not result2["scale_recovered"])),
        "scale_warning": "translation values are not metrically reliable because at least one camera used unit SLAM scale"
        if (not result1["scale_recovered"]) or (not result2["scale_recovered"])
        else None,
        "camera1_timestamp_source": result1["trajectory_timestamp_source"],
        "camera2_timestamp_source": result2["trajectory_timestamp_source"],
        "num_camera2_keyframes": int(len(ts2)),
        "num_skipped_out_of_range": int(skipped_out_of_range),
        "num_skipped_bracket_width": int(skipped_bracket),
        "num_skipped_anchor_distance_or_time": int(skipped_distance),
        "num_matches_used": int(len(matches)),
        "max_bracket_width_s": args.max_bracket_width_s,
        "max_distance_from_anchor_m": args.max_distance_from_anchor_m,
        "camera1_max_distance_from_anchor_m": camera1_distance_cutoff,
        "camera2_max_distance_from_anchor_m": camera2_distance_cutoff,
        "max_time_from_anchor_s": args.max_time_from_anchor_s,
        "camera1_max_time_from_anchor_s": camera1_time_cutoff,
        "camera2_max_time_from_anchor_s": camera2_time_cutoff,
        "rotation_deviation_from_mean_deg": {
            "median": float(np.median(angle_devs)),
            "mean": float(angle_devs.mean()),
            "max": float(angle_devs.max()),
        },
        "translation_deviation_from_median_m": {
            "median": float(np.median(trans_devs)),
            "mean": float(trans_devs.mean()),
            "max": float(trans_devs.max()),
        },
        "optimization": opt_diag,
        "optimization_grouped": grp_diag,
    }
    plot_matches = {
        "camera1_positions": np.array([m["camera1_position"] for m in matches], dtype=np.float64),
        "camera2_positions": np.array([m["camera2_position"] for m in matches], dtype=np.float64),
        "timestamps": np.array([m["timestamp"] for m in matches], dtype=np.float64),
    }
    return extrinsic_yaml, diagnostics, plot_matches


def main() -> int:
    args = parse_args()

    run_dir = args.camera1_raw_csv.parent
    alignment_dir = run_dir / "aruco_alignment"
    pair = f"{args.camera1_name}_{args.camera2_name}"
    output_plot = args.output_plot or alignment_dir / f"{pair}_trajectories.png"
    output_alignment_json = args.output_alignment_json or alignment_dir / f"{pair}_alignment.json"
    output_extrinsic_yaml = args.output_extrinsic_yaml or alignment_dir / f"{pair}_extrinsic.yaml"

    rows1 = read_points_csv(args.camera1_points_csv)
    rows2 = read_points_csv(args.camera2_points_csv)
    calib1 = read_calibration(args.camera1_config)
    calib2 = read_calibration(args.camera2_config)

    marker_id = choose_marker_id(rows1, rows2, args.camera1_name, args.camera2_name, args.marker_id)
    use_source_timestamps = not args.raw_ros_timestamps
    if args.alignment_source == "visual-aruco-sim3":
        result1 = align_camera_visual_aruco_sim3(
            args.camera1_name,
            marker_id,
            args.camera1_raw_csv,
            calib1,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            use_source_timestamps,
        )
        result2 = align_camera_visual_aruco_sim3(
            args.camera2_name,
            marker_id,
            args.camera2_raw_csv,
            calib2,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            use_source_timestamps,
        )
    elif args.alignment_source == "visual-aruco-motion-scale":
        result1 = align_camera_visual_aruco_motion_scale(
            args.camera1_name,
            marker_id,
            args.camera1_raw_csv,
            calib1,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            args.visual_aruco_motion_min_distance_m,
            args.visual_aruco_motion_max_pairs,
            use_source_timestamps,
        )
        result2 = align_camera_visual_aruco_motion_scale(
            args.camera2_name,
            marker_id,
            args.camera2_raw_csv,
            calib2,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            args.visual_aruco_motion_min_distance_m,
            args.visual_aruco_motion_max_pairs,
            use_source_timestamps,
        )
    elif args.alignment_source == "optimized-anchor":
        if args.camera1_scale is None or args.camera2_scale is None:
            raise SystemExit(
                "--alignment-source optimized-anchor requires both --camera1-scale and --camera2-scale "
                "(it fits rotation from many keyframes but always takes scale externally)."
            )
        result1 = align_camera_optimized_anchor(
            args.camera1_name,
            marker_id,
            args.camera1_raw_csv,
            calib1,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            args.camera1_scale,
            use_source_timestamps,
        )
        result2 = align_camera_optimized_anchor(
            args.camera2_name,
            marker_id,
            args.camera2_raw_csv,
            calib2,
            args.marker_length_m,
            args.dictionary_size,
            args.dictionary_bits,
            args.visual_aruco_min_detections,
            args.visual_aruco_max_rms_px,
            args.visual_aruco_max_keyframes,
            args.visual_aruco_trim_mad,
            args.camera2_scale,
            use_source_timestamps,
        )
    else:
        raise SystemExit(f"Unknown --alignment-source: {args.alignment_source}")

    # --- Stage 2 outputs ---
    extrinsic_yaml, diagnostics, plot_matches = compute_extrinsic(args, marker_id, result1, result2)

    # --- Stage 1 outputs ---
    plot_trajectories(
        [result1, result2],
        marker_id,
        args.marker_length_m,
        output_plot,
        args.show,
        max_distance_from_anchor_m=args.max_distance_from_anchor_m,
        per_camera_max_distance_from_anchor_m=[camera_distance_cutoff(args, 1), camera_distance_cutoff(args, 2)]
        if (camera_distance_cutoff(args, 1) is not None or camera_distance_cutoff(args, 2) is not None)
        else None,
        max_time_from_anchor_s=args.max_time_from_anchor_s,
        per_camera_max_time_from_anchor_s=[camera_time_cutoff(args, 1), camera_time_cutoff(args, 2)]
        if (camera_time_cutoff(args, 1) is not None or camera_time_cutoff(args, 2) is not None)
        else None,
        surviving_match_points=[plot_matches["camera1_positions"], plot_matches["camera2_positions"]],
        show_anchor_marker=not args.hide_anchor_marker,
    )

    alignment_summary = {
        "marker_id": marker_id,
        "cameras": [
            {
                "name": res["name"],
                "keyframe_id": res["kf_id"],
                "num_marker_points_used": res["num_marker_points"],
                "num_visual_corners_used": res["num_visual_corners"],
                "marker_pnp_input": res["marker_pnp_input"],
                "visual_marker_keyframes": res["visual_marker_keyframes"],
                "visual_marker_detections": res["visual_marker_detections"],
                "anchor_image_path": res["anchor_image_path"],
                "marker_pnp_reprojection_rms_px": res["marker_pnp_rms_px"],
                "scale_m_per_slam_unit": res["scale"],
                "scale_source": res["scale_source"],
                "scale_recovered": res["scale_recovered"],
                "scale_warning": res["scale_warning"],
                "timestamp_source": res["trajectory_timestamp_source"],
                "rotation_slam_to_marker": res["R_slam_to_marker"].tolist(),
                "translation_slam_to_marker_m": res["t_slam_to_marker"].tolist(),
                "visual_sim3_diagnostics": res.get("visual_sim3_diagnostics"),
                "visual_motion_scale_diagnostics": res.get("visual_motion_scale_diagnostics"),
                "trajectory_num_keyframes": int(res["trajectory"].shape[0]),
            }
            for res in (result1, result2)
        ],
    }
    output_alignment_json.parent.mkdir(parents=True, exist_ok=True)
    with output_alignment_json.open("w") as handle:
        json.dump(alignment_summary, handle, indent=2)
    print(f"Saved alignment summary: {output_alignment_json}")

    output_extrinsic_yaml.parent.mkdir(parents=True, exist_ok=True)
    def yaml_scalar(value: object) -> str:
        return "null" if value is None else str(value)

    with output_extrinsic_yaml.open("w") as handle:
        handle.write(extrinsic_yaml)
        handle.write("\ncalibration_diagnostics:\n")
        for key, value in diagnostics.items():
            if isinstance(value, dict):
                handle.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    handle.write(f"    {sub_key}: {yaml_scalar(sub_value)}\n")
            else:
                handle.write(f"  {key}: {yaml_scalar(value)}\n")
    print(f"Saved: {output_extrinsic_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
