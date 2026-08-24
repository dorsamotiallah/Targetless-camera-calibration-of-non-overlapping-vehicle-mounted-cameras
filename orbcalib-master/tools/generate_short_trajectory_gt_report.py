#!/usr/bin/env python3
"""Create a one-page-per-dataset PDF comparing short trajectory results to GT."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


DATASETS = [
    "lingonberries",
    "mobilelab-fenced",
    "mobilelab-logs",
    "rakennustalo-fence",
    "rakennustalo-outside-fence",
    "fablabdoor_bigger_marker",
]
PAIRS = [("front", "back"), ("front", "left"), ("front", "right")]
# fablabdoor_bigger_marker is the same physical rig, but additionally has a
# left-right pair (no separate front leg was recorded for that combination).
DATASET_PAIRS = {
    "fablabdoor_bigger_marker": PAIRS + [("left", "right")],
}
METHODS = [
    ("Ground", "calibration_ground_scale"),
    ("Sim3", "calibration_sim3_scale"),
    ("PointPair", "calibration_aruco_points_scale"),
]
# Diagnostic-only variants (not part of the main 3-method comparison/CSV):
#   Sim3Scale: Sim3's recovered per-camera scale value reused with the anchor
#     (single-keyframe) rotation pipeline instead of Sim3's own rotation fit.
#   GroundSim3Rot / PointPairSim3Rot: Sim3's robust multi-keyframe ROTATION fit,
#     but with Ground's / PointPair's scale value substituted in instead of
#     Sim3's own scale.
# Toggle DIAGNOSTIC_METHODS on to include them in an ad-hoc analysis run.
DIAGNOSTIC_METHODS = [
    ("Sim3Scale", "calibration_sim3scale_anchor"),
    ("GroundSim3Rot", "calibration_groundscale_sim3rot"),
    ("PointPairSim3Rot", "calibration_pointpairscale_sim3rot"),
]
# Older runs (fablabdoor_bigger_marker) saved the Ground-scale calibration under
# "calibration_ground" instead of "calibration_ground_scale". Try these aliases
# in order when the primary method_dir doesn't exist.
METHOD_DIR_ALIASES = {
    "calibration_ground_scale": ["calibration_ground_scale", "calibration_ground"],
}
ESTIMATION_METHODS = [
    ("averaged", "T_{pair}"),
    ("optimized", "T_{pair}_optimized"),
    ("optimized_grouped", "T_{pair}_optimized_grouped"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results_fisheye/Shorter_trajectories"),
        help="Root containing dataset/pair result folders.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("../Agilex Recordings/Intrinsic_ground_truths/robot_relative_extrinsics.yaml"),
        help="Agilex robot_relative_extrinsics.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_fisheye/Shorter_trajectories/short_trajectories_gt_comparison.pdf"),
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV summary output.")
    return parser.parse_args()


def read_matrix_block(text: str, key: str) -> Optional[np.ndarray]:
    marker = f"{key}:"
    start = text.find(marker)
    if start < 0:
        return None
    next_key = re.search(r"\n[A-Za-z0-9_]+:", text[start + len(marker) :])
    end = len(text) if next_key is None else start + len(marker) + next_key.start()
    block = text[start:end]
    rows = []
    in_matrix = False
    for line in block.splitlines():
        if line.strip() == "matrix:":
            in_matrix = True
            continue
        if not in_matrix:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ["):
            if rows:
                break
            continue
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stripped)]
        rows.append(values)
        if len(rows) == 4:
            break
    if len(rows) != 4:
        return None
    return np.array(rows, dtype=float)


def load_gt(path: Path) -> Dict[str, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    gt = {}
    all_pairs = set(PAIRS)
    for extra in DATASET_PAIRS.values():
        all_pairs.update(extra)
    for cam1, cam2 in all_pairs:
        key = f"T_{cam1}_{cam2}"
        mat = read_matrix_block(text, key)
        if mat is None:
            raise RuntimeError(f"Could not parse {key} from {path}")
        gt[key] = mat
    return gt


def rotation_error_deg(r_est: np.ndarray, r_gt: np.ndarray) -> float:
    delta = r_est @ r_gt.T
    cos_theta = (np.trace(delta) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0))))


def rotation_vector(r_mat: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(r_mat) - 1.0) / 2.0
    theta = math.acos(float(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
    denom = 2.0 * math.sin(theta)
    axis = np.array(
        [
            r_mat[2, 1] - r_mat[1, 2],
            r_mat[0, 2] - r_mat[2, 0],
            r_mat[1, 0] - r_mat[0, 1],
        ],
        dtype=float,
    ) / denom
    return axis * theta


def safe_get(data: object, keys: Iterable[object]) -> object:
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def load_alignment_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stats = {}
    diagnostics = data.get("extrinsic_diagnostics") or data.get("calibration_diagnostics") or data
    stats["matches"] = diagnostics.get("num_matches_used")
    stats["marker"] = data.get("marker_id") or diagnostics.get("marker_id")
    cameras = data.get("cameras")
    if isinstance(cameras, list):
        scales = []
        for cam in cameras:
            name = cam.get("name")
            scale = cam.get("scale_m_per_slam_unit")
            if name is not None and scale is not None:
                scales.append(f"{name}:{float(scale):.3g}")
        stats["scales"] = ", ".join(scales)
    return stats


def yaml_scalar(text: str, key: str) -> Optional[str]:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value in ("null", "None", "") else value


def result_for_matrix(
    *,
    mat: np.ndarray,
    gt: np.ndarray,
    pair: str,
    scale_method: str,
    estimation_method: str,
    extrinsic_path: Path,
    alignment_path: Path,
    text: str,
) -> Dict[str, object]:
    t_est = mat[:3, 3]
    t_gt = gt[:3, 3]
    dt = t_est - t_gt
    row: Dict[str, object] = {
        "pair": pair,
        "scale_method": scale_method,
        "estimation_method": estimation_method,
        "path": extrinsic_path,
        "status": "ok",
        "t_est": t_est,
        "t_gt": t_gt,
        "rvec_est": rotation_vector(mat[:3, :3]),
        "rvec_gt": rotation_vector(gt[:3, :3]),
        "dt": dt,
        "trans_err": float(np.linalg.norm(dt)),
        "rot_err": rotation_error_deg(mat[:3, :3], gt[:3, :3]),
    }
    row.update(load_alignment_stats(alignment_path))
    matches = yaml_scalar(text, "num_matches_used")
    if matches is not None:
        try:
            row["matches"] = int(matches)
        except ValueError:
            row["matches"] = matches
    marker_id = yaml_scalar(text, "marker_id")
    if marker_id is not None:
        try:
            row["marker"] = int(marker_id)
        except ValueError:
            row["marker"] = marker_id
    return row


def resolve_method_dir(run_dir: Path, method_dir: str) -> str:
    for candidate in METHOD_DIR_ALIASES.get(method_dir, [method_dir]):
        if (run_dir / candidate).is_dir():
            return candidate
    return method_dir


def result_candidates(run_dir: Path, scale_method: str, method_dir: str, cam1: str, cam2: str, gt: np.ndarray) -> List[Dict[str, object]]:
    pair = f"{cam1}_{cam2}"
    method_dir = resolve_method_dir(run_dir, method_dir)
    align_dir = run_dir / method_dir / "aruco_alignment"
    extrinsic_path = align_dir / f"{pair}_extrinsic.yaml"
    alignment_path = align_dir / f"{pair}_alignment.json"
    if not extrinsic_path.exists():
        return []

    text = extrinsic_path.read_text(encoding="utf-8")
    candidates = []
    for estimation_method, key_template in ESTIMATION_METHODS:
        key = key_template.format(pair=pair)
        mat = read_matrix_block(text, key)
        if mat is None:
            continue
        candidates.append(
            result_for_matrix(
                mat=mat,
                gt=gt,
                pair=pair,
                scale_method=scale_method,
                estimation_method=estimation_method,
                extrinsic_path=extrinsic_path,
                alignment_path=alignment_path,
                text=text,
            )
        )
    return candidates


def missing_status(run_dir: Path, method_dir: str, cam1: str, cam2: str) -> str:
    pair = f"{cam1}_{cam2}"
    method_dir = resolve_method_dir(run_dir, method_dir)
    align_dir = run_dir / method_dir / "aruco_alignment"
    extrinsic_path = align_dir / f"{pair}_extrinsic.yaml"
    if extrinsic_path.exists():
        return "unparseable"
    log_path = align_dir / f"{pair}_runner.log"
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        return tail[0][:90] if tail else "missing extrinsic"
    return "missing"


def pick_best(candidates: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    by_trans = sorted(candidates, key=lambda row: (float(row["trans_err"]), float(row["rot_err"])))
    by_rot = sorted(candidates, key=lambda row: (float(row["rot_err"]), float(row["trans_err"])))
    trans_rank = {id(row): rank for rank, row in enumerate(by_trans, start=1)}
    rot_rank = {id(row): rank for rank, row in enumerate(by_rot, start=1)}
    for row in candidates:
        row["translation_rank"] = trans_rank[id(row)]
        row["rotation_rank"] = rot_rank[id(row)]
        row["combined_rank"] = trans_rank[id(row)] + rot_rank[id(row)]
        row["best_in_both_metrics"] = trans_rank[id(row)] == 1 and rot_rank[id(row)] == 1
    return min(
        candidates,
        key=lambda row: (
            int(row["combined_rank"]),
            float(row["trans_err"]),
            float(row["rot_err"]),
            str(row["scale_method"]),
            str(row["estimation_method"]),
        ),
    )


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in vec) + "]"


def fmt_vec_compact(vec: np.ndarray) -> str:
    return ",".join(f"{v:+.3f}" for v in vec)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def add_table(ax: plt.Axes, rows: List[List[str]], col_labels: List[str], bbox: List[float], fontsize: int = 7) -> None:
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left", loc="upper left", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        if r == 0:
            cell.set_facecolor("#e9edf4")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f8fa")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    import csv

    fields = [
        "dataset",
        "pair",
        "scale_method",
        "estimation_method",
        "selected_best",
        "best_in_both_metrics",
        "translation_rank",
        "rotation_rank",
        "combined_rank",
        "status",
        "rotation_error_deg",
        "translation_error_m",
        "translation_error_xyz_m",
        "translation_est_xyz_m",
        "translation_gt_xyz_m",
        "rotation_est_rvec_rad",
        "rotation_gt_rvec_rad",
        "matches",
        "marker",
        "scales",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            for key in (
                "translation_error_xyz_m",
                "translation_est_xyz_m",
                "translation_gt_xyz_m",
                "rotation_est_rvec_rad",
                "rotation_gt_rvec_rad",
            ):
                value = out.get(key)
                if isinstance(value, np.ndarray):
                    out[key] = " ".join(f"{x:.9g}" for x in value)
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    gt_path = args.ground_truth.resolve()
    output_path = args.output.resolve()
    csv_path = args.csv.resolve() if args.csv else output_path.with_suffix(".csv")

    gt = load_gt(gt_path)
    all_rows: List[Dict[str, object]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for dataset in DATASETS:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.text(0.02, 0.965, dataset, fontsize=18, weight="bold", transform=ax.transAxes)
            ax.text(
                0.02,
                0.935,
                "Best row per T_front_<camera> selected across scale method and estimation method. "
                "Best = lowest combined rank over translation and rotation errors.",
                fontsize=8.2,
                transform=ax.transAxes,
            )
            ax.text(0.02, 0.91, f"GT: {gt_path}", fontsize=7.2, color="#555", transform=ax.transAxes)

            table_rows: List[List[str]] = []
            for cam1, cam2 in DATASET_PAIRS.get(dataset, PAIRS):
                run_dir = root / dataset / f"{cam1}_{cam2}"
                gt_mat = gt[f"T_{cam1}_{cam2}"]
                candidates: List[Dict[str, object]] = []
                missing_messages = []
                for method_name, method_dir in METHODS:
                    method_candidates = result_candidates(run_dir, method_name, method_dir, cam1, cam2, gt_mat)
                    if method_candidates:
                        candidates.extend(method_candidates)
                    else:
                        missing_messages.append(f"{method_name}: {missing_status(run_dir, method_dir, cam1, cam2)}")
                best = pick_best(candidates)
                for row in candidates:
                    record = {
                        "dataset": dataset,
                        "pair": f"{cam1}_{cam2}",
                        "scale_method": row.get("scale_method"),
                        "estimation_method": row.get("estimation_method"),
                        "selected_best": row is best,
                        "best_in_both_metrics": row.get("best_in_both_metrics"),
                        "translation_rank": row.get("translation_rank"),
                        "rotation_rank": row.get("rotation_rank"),
                        "combined_rank": row.get("combined_rank"),
                        "status": row["status"],
                        "rotation_error_deg": row.get("rot_err"),
                        "translation_error_m": row.get("trans_err"),
                        "translation_error_xyz_m": row.get("dt"),
                        "translation_est_xyz_m": row.get("t_est"),
                        "translation_gt_xyz_m": row.get("t_gt"),
                        "rotation_est_rvec_rad": row.get("rvec_est"),
                        "rotation_gt_rvec_rad": row.get("rvec_gt"),
                        "matches": row.get("matches"),
                        "marker": row.get("marker"),
                        "scales": row.get("scales"),
                        "source": str(row.get("path", "")),
                    }
                    all_rows.append(record)
                if best is not None:
                    note = (
                        "best both (t1/r1)"
                        if best.get("best_in_both_metrics")
                        else f"t{best.get('translation_rank')}/r{best.get('rotation_rank')} sum{best.get('combined_rank')}"
                    )
                    table_rows.append(
                        [
                            f"{cam1}-{cam2}",
                            str(best["scale_method"]),
                            str(best["estimation_method"]),
                            fmt_vec_compact(best["t_est"]),
                            fmt_vec_compact(best["rvec_est"]),
                            fmt_vec_compact(best["t_gt"]),
                            fmt_vec_compact(best["rvec_gt"]),
                            fmt_vec_compact(best["dt"]),
                            fmt(best.get("trans_err"), 3),
                            fmt(best.get("rot_err"), 2),
                            fmt(best.get("matches"), 0),
                            note,
                        ]
                    )
                else:
                    table_rows.append(
                        [
                            f"{cam1}-{cam2}",
                            "missing",
                            "; ".join(missing_messages) or "missing",
                            "-",
                            "-",
                            fmt_vec_compact(gt_mat[:3, 3]),
                            fmt_vec_compact(rotation_vector(gt_mat[:3, :3])),
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                        ]
                    )

            add_table(
                ax,
                table_rows,
                [
                    "Pair",
                    "Scale",
                    "Estimator",
                    "Est t xyz [m]",
                    "Est rvec xyz [rad]",
                    "GT t xyz [m]",
                    "GT rvec xyz [rad]",
                    "Est-GT t [m]",
                    "|dt| m",
                    "Rot err deg",
                    "Matches",
                    "Pick",
                ],
                [0.01, 0.25, 0.98, 0.57],
                fontsize=5.6,
            )

            notes = [
                "Scale: Ground = recovered ground-plane scale; Sim3 = visual ArUco Sim3 scale/alignment.",
                "Estimator: averaged = T_front_camera, optimized = robust SE(3), optimized_grouped = grouped robust SE(3).",
                "Pick is 'best both' when the same candidate has the lowest translation and rotation error; otherwise rank is translation-rank + rotation-rank.",
                "Rotation vector is axis-angle/Rodrigues form in radians.",
                "Missing entries mean the corresponding extrinsic YAML was not produced.",
            ]
            ax.text(0.02, 0.15, "\n".join(notes), fontsize=7.2, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    write_csv(csv_path, all_rows)
    print(f"Wrote PDF: {output_path}")
    print(f"Wrote CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
