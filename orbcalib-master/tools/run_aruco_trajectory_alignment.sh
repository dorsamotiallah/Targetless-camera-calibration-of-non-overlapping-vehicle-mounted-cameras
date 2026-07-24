#!/usr/bin/env bash
set -euo pipefail

# Run ArUco point/scale extraction for two cameras, then align their
# trajectories through the shared marker frame.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
RESULTS_ROOT=${RESULTS_ROOT:-"$REPO_DIR/results_agilex_outdoor_latest"}

RUN_ID=${RUN_ID:-}
RUN_DIR=${RUN_DIR:-}
CAMERA1_RUN_DIR=${CAMERA1_RUN_DIR:-}
CAMERA2_RUN_DIR=${CAMERA2_RUN_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
CAMERA1_NAME=${CAMERA1_NAME:-front}
CAMERA2_NAME=${CAMERA2_NAME:-left}
CAMERA1_CONFIG=${CAMERA1_CONFIG:-}
CAMERA2_CONFIG=${CAMERA2_CONFIG:-}
MARKER_ID=${MARKER_ID:-0}
MARKER_LENGTH_M=${MARKER_LENGTH_M:-0.182}
DICTIONARY_SIZE=${DICTIONARY_SIZE:-50}
KEYFRAME_CANDIDATES=${KEYFRAME_CANDIDATES:-5}
MAX_DISTANCE_FROM_ANCHOR_M=${MAX_DISTANCE_FROM_ANCHOR_M:-}
CAMERA1_MAX_DISTANCE_FROM_ANCHOR_M=${CAMERA1_MAX_DISTANCE_FROM_ANCHOR_M:-}
CAMERA2_MAX_DISTANCE_FROM_ANCHOR_M=${CAMERA2_MAX_DISTANCE_FROM_ANCHOR_M:-}
MAX_TIME_FROM_ANCHOR_S=${MAX_TIME_FROM_ANCHOR_S:-}
CAMERA1_MAX_TIME_FROM_ANCHOR_S=${CAMERA1_MAX_TIME_FROM_ANCHOR_S:-}
CAMERA2_MAX_TIME_FROM_ANCHOR_S=${CAMERA2_MAX_TIME_FROM_ANCHOR_S:-}
MAX_BRACKET_WIDTH_S=${MAX_BRACKET_WIDTH_S:-}
SHOW=0
DEBUG_IMAGES=1
RAW_ROS_TIMESTAMPS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") (--run-id NAME | --run-dir PATH) --camera1 NAME --camera2 NAME [options]

Runs, in order:
  1. estimate_aruco_slam_scale.py for camera1
  2. estimate_aruco_slam_scale.py for camera2
  3. align_trajectories_to_aruco.py using the generated point CSVs and any
     finite recommended_scale_m_per_slam_unit values from the scale summaries.

Options:
  --run-id NAME                 Run folder name under --results-root.
  --run-dir PATH                Full run folder path. Overrides --run-id.
  --camera1-run-dir PATH        Run folder containing camera1 raw CSV/manifest.
                                Defaults to --run-dir.
  --camera2-run-dir PATH        Run folder containing camera2 raw CSV/manifest.
                                Defaults to --run-dir.
  --output-dir PATH             Folder for aruco_scale and aruco_alignment outputs.
                                Defaults to --run-dir.
  --results-root PATH           Default: $RESULTS_ROOT
  --camera1 NAME                First camera name. Default: $CAMERA1_NAME
  --camera2 NAME                Second camera name. Default: $CAMERA2_NAME
  --camera1-config PATH         Default: config/sim/agilex_<camera1>_defished_cam.yaml
  --camera2-config PATH         Default: config/sim/agilex_<camera2>_defished_cam.yaml
  --marker-id ID                Marker ID to use. Default: $MARKER_ID
  --marker-length-m M           ArUco marker side length in meters. Default: $MARKER_LENGTH_M
  --dictionary-size N           OpenCV 6x6 dictionary size. Default: $DICTIONARY_SIZE
  --keyframe-candidates N       Alignment keyframe candidates. Default: $KEYFRAME_CANDIDATES
  --max-distance-from-anchor-m M
  --camera1-max-distance-from-anchor-m M
  --camera2-max-distance-from-anchor-m M
  --max-time-from-anchor-s S
  --camera1-max-time-from-anchor-s S
  --camera2-max-time-from-anchor-s S
  --max-bracket-width-s S
  --show                        Show the saved alignment plot interactively.
  --no-debug-images             Skip ArUco scale debug overlay images.
  --raw-ros-timestamps          Use raw replay timestamps instead of source PNG timestamps.
  -h, --help                    Show this help.

Environment overrides are also supported for the uppercase variable names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --camera1-run-dir) CAMERA1_RUN_DIR="$2"; shift 2 ;;
    --camera2-run-dir) CAMERA2_RUN_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --camera1) CAMERA1_NAME="$2"; shift 2 ;;
    --camera2) CAMERA2_NAME="$2"; shift 2 ;;
    --camera1-config) CAMERA1_CONFIG="$2"; shift 2 ;;
    --camera2-config) CAMERA2_CONFIG="$2"; shift 2 ;;
    --marker-id) MARKER_ID="$2"; shift 2 ;;
    --marker-length-m) MARKER_LENGTH_M="$2"; shift 2 ;;
    --dictionary-size) DICTIONARY_SIZE="$2"; shift 2 ;;
    --keyframe-candidates) KEYFRAME_CANDIDATES="$2"; shift 2 ;;
    --max-distance-from-anchor-m) MAX_DISTANCE_FROM_ANCHOR_M="$2"; shift 2 ;;
    --camera1-max-distance-from-anchor-m) CAMERA1_MAX_DISTANCE_FROM_ANCHOR_M="$2"; shift 2 ;;
    --camera2-max-distance-from-anchor-m) CAMERA2_MAX_DISTANCE_FROM_ANCHOR_M="$2"; shift 2 ;;
    --max-time-from-anchor-s) MAX_TIME_FROM_ANCHOR_S="$2"; shift 2 ;;
    --camera1-max-time-from-anchor-s) CAMERA1_MAX_TIME_FROM_ANCHOR_S="$2"; shift 2 ;;
    --camera2-max-time-from-anchor-s) CAMERA2_MAX_TIME_FROM_ANCHOR_S="$2"; shift 2 ;;
    --max-bracket-width-s) MAX_BRACKET_WIDTH_S="$2"; shift 2 ;;
    --show) SHOW=1; shift ;;
    --no-debug-images) DEBUG_IMAGES=0; shift ;;
    --raw-ros-timestamps) RAW_ROS_TIMESTAMPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  if [[ -z "$RUN_ID" ]]; then
    echo "Missing --run-id or --run-dir." >&2
    usage >&2
    exit 2
  fi
  RUN_DIR="$RESULTS_ROOT/$RUN_ID"
fi

if [[ ! "$RUN_DIR" = /* ]]; then
  RUN_DIR="$REPO_DIR/$RUN_DIR"
fi

CAMERA1_RUN_DIR=${CAMERA1_RUN_DIR:-"$RUN_DIR"}
CAMERA2_RUN_DIR=${CAMERA2_RUN_DIR:-"$RUN_DIR"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RUN_DIR"}

if [[ ! "$CAMERA1_RUN_DIR" = /* ]]; then
  CAMERA1_RUN_DIR="$REPO_DIR/$CAMERA1_RUN_DIR"
fi
if [[ ! "$CAMERA2_RUN_DIR" = /* ]]; then
  CAMERA2_RUN_DIR="$REPO_DIR/$CAMERA2_RUN_DIR"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_DIR="$REPO_DIR/$OUTPUT_DIR"
fi

if [[ -z "$CAMERA1_CONFIG" ]]; then
  CAMERA1_CONFIG="$REPO_DIR/config/sim/agilex_${CAMERA1_NAME}_defished_cam.yaml"
elif [[ ! "$CAMERA1_CONFIG" = /* ]]; then
  CAMERA1_CONFIG="$REPO_DIR/$CAMERA1_CONFIG"
fi

if [[ -z "$CAMERA2_CONFIG" ]]; then
  CAMERA2_CONFIG="$REPO_DIR/config/sim/agilex_${CAMERA2_NAME}_defished_cam.yaml"
elif [[ ! "$CAMERA2_CONFIG" = /* ]]; then
  CAMERA2_CONFIG="$REPO_DIR/$CAMERA2_CONFIG"
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

LOG_DIR="$OUTPUT_DIR/aruco_alignment"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${CAMERA1_NAME}_${CAMERA2_NAME}_runner.log"
exec > >(tee "$LOG_FILE") 2>&1

json_optional_number() {
  local path="$1"
  local key="$2"
  python3 - "$path" "$key" <<'PY'
import json
import math
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    value = json.load(handle).get(key)
if value is None:
    raise SystemExit(0)
if not isinstance(value, (int, float)) or not math.isfinite(value):
    raise SystemExit(f"{path} contains a non-numeric {key}: {value!r}")
print(f"{value:.17g}")
PY
}

scale_camera() {
  local camera="$1"
  local camera_run_dir="$2"
  local out_dir="$3"
  local scale_args=(
    python3 tools/estimate_aruco_slam_scale.py
    --run-dir "$camera_run_dir"
    --camera "$camera"
    --marker-length-m "$MARKER_LENGTH_M"
    --dictionary-size "$DICTIONARY_SIZE"
    --marker-id "$MARKER_ID"
    --out-dir "$out_dir"
  )
  if [[ "$DEBUG_IMAGES" != "1" ]]; then
    scale_args+=(--no-debug-images)
  fi

  echo "Estimating ArUco scale for $camera..."
  "${scale_args[@]}"
}

cd "$REPO_DIR"

require_dir "$RUN_DIR"
require_dir "$CAMERA1_RUN_DIR"
require_dir "$CAMERA2_RUN_DIR"
mkdir -p "$OUTPUT_DIR"
require_file "$CAMERA1_CONFIG"
require_file "$CAMERA2_CONFIG"
require_file "$CAMERA1_RUN_DIR/${CAMERA1_NAME}_raw_keyframe_observations.csv"
require_file "$CAMERA2_RUN_DIR/${CAMERA2_NAME}_raw_keyframe_observations.csv"

echo "Logging to: $LOG_FILE"
scale_camera "$CAMERA1_NAME" "$CAMERA1_RUN_DIR" "$OUTPUT_DIR/aruco_scale/$CAMERA1_NAME"
scale_camera "$CAMERA2_NAME" "$CAMERA2_RUN_DIR" "$OUTPUT_DIR/aruco_scale/$CAMERA2_NAME"

CAMERA1_POINTS="$OUTPUT_DIR/aruco_scale/$CAMERA1_NAME/${CAMERA1_NAME}_aruco_scale_points.csv"
CAMERA2_POINTS="$OUTPUT_DIR/aruco_scale/$CAMERA2_NAME/${CAMERA2_NAME}_aruco_scale_points.csv"
CAMERA1_SUMMARY="$OUTPUT_DIR/aruco_scale/$CAMERA1_NAME/${CAMERA1_NAME}_aruco_scale_summary.json"
CAMERA2_SUMMARY="$OUTPUT_DIR/aruco_scale/$CAMERA2_NAME/${CAMERA2_NAME}_aruco_scale_summary.json"

require_file "$CAMERA1_POINTS"
require_file "$CAMERA2_POINTS"
require_file "$CAMERA1_SUMMARY"
require_file "$CAMERA2_SUMMARY"

CAMERA1_SCALE=$(json_optional_number "$CAMERA1_SUMMARY" recommended_scale_m_per_slam_unit)
CAMERA2_SCALE=$(json_optional_number "$CAMERA2_SUMMARY" recommended_scale_m_per_slam_unit)

align_args=(
  python3 tools/align_trajectories_to_aruco.py
  --camera1-name "$CAMERA1_NAME"
  --camera1-points-csv "$CAMERA1_POINTS"
  --camera1-raw-csv "$CAMERA1_RUN_DIR/${CAMERA1_NAME}_raw_keyframe_observations.csv"
  --camera1-config "$CAMERA1_CONFIG"
  --camera2-name "$CAMERA2_NAME"
  --camera2-points-csv "$CAMERA2_POINTS"
  --camera2-raw-csv "$CAMERA2_RUN_DIR/${CAMERA2_NAME}_raw_keyframe_observations.csv"
  --camera2-config "$CAMERA2_CONFIG"
  --marker-id "$MARKER_ID"
  --marker-length-m "$MARKER_LENGTH_M"
  --dictionary-size "$DICTIONARY_SIZE"
  --keyframe-candidates "$KEYFRAME_CANDIDATES"
  --output-plot "$OUTPUT_DIR/aruco_alignment/${CAMERA1_NAME}_${CAMERA2_NAME}_trajectories.png"
  --output-alignment-json "$OUTPUT_DIR/aruco_alignment/${CAMERA1_NAME}_${CAMERA2_NAME}_alignment.json"
  --output-extrinsic-yaml "$OUTPUT_DIR/aruco_alignment/${CAMERA1_NAME}_${CAMERA2_NAME}_extrinsic.yaml"
)

if [[ -n "$CAMERA1_SCALE" ]]; then
  echo "Using $CAMERA1_NAME scale: $CAMERA1_SCALE m/slam-unit"
  align_args+=(--camera1-scale "$CAMERA1_SCALE")
else
  echo "$CAMERA1_NAME scale unavailable from SLAM marker points; alignment may use visual ArUco fallback with unit scale."
fi
if [[ -n "$CAMERA2_SCALE" ]]; then
  echo "Using $CAMERA2_NAME scale: $CAMERA2_SCALE m/slam-unit"
  align_args+=(--camera2-scale "$CAMERA2_SCALE")
else
  echo "$CAMERA2_NAME scale unavailable from SLAM marker points; alignment may use visual ArUco fallback with unit scale."
fi

if [[ -n "$MAX_DISTANCE_FROM_ANCHOR_M" ]]; then
  align_args+=(--max-distance-from-anchor-m "$MAX_DISTANCE_FROM_ANCHOR_M")
fi
if [[ -n "$CAMERA1_MAX_DISTANCE_FROM_ANCHOR_M" ]]; then
  align_args+=(--camera1-max-distance-from-anchor-m "$CAMERA1_MAX_DISTANCE_FROM_ANCHOR_M")
fi
if [[ -n "$CAMERA2_MAX_DISTANCE_FROM_ANCHOR_M" ]]; then
  align_args+=(--camera2-max-distance-from-anchor-m "$CAMERA2_MAX_DISTANCE_FROM_ANCHOR_M")
fi
if [[ -n "$MAX_TIME_FROM_ANCHOR_S" ]]; then
  align_args+=(--max-time-from-anchor-s "$MAX_TIME_FROM_ANCHOR_S")
fi
if [[ -n "$CAMERA1_MAX_TIME_FROM_ANCHOR_S" ]]; then
  align_args+=(--camera1-max-time-from-anchor-s "$CAMERA1_MAX_TIME_FROM_ANCHOR_S")
fi
if [[ -n "$CAMERA2_MAX_TIME_FROM_ANCHOR_S" ]]; then
  align_args+=(--camera2-max-time-from-anchor-s "$CAMERA2_MAX_TIME_FROM_ANCHOR_S")
fi
if [[ -n "$MAX_BRACKET_WIDTH_S" ]]; then
  align_args+=(--max-bracket-width-s "$MAX_BRACKET_WIDTH_S")
fi
if [[ "$SHOW" == "1" ]]; then
  align_args+=(--show)
fi
if [[ "$RAW_ROS_TIMESTAMPS" == "1" ]]; then
  align_args+=(--raw-ros-timestamps)
fi

echo "Aligning trajectories..."
"${align_args[@]}"

echo "Done. Outputs are under: $OUTPUT_DIR"
