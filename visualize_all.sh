#!/usr/bin/env bash
#
# ==============================================================================
# Post-training Visualization Pipeline
# ==============================================================================
# Run all visualizations for a trained federated change detection model.
#
# This script finds the latest model checkpoint (or uses a user-specified one),
# runs the evaluation + visualization tool, parses training logs for metric
# curves, and collects every artifact into a single output directory.
# ==============================================================================

# ==============================================================================
# Script metadata
# ==============================================================================

readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DESCRIPTION="Post-training visualization pipeline for FederatedRSCD"

readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly SCRIPT_NAME="${SCRIPT_PATH##*/}"

fct_get_script_dir() {
    local source="${SCRIPT_PATH}"
    local dir="${source%/*}"
    if [[ "${dir}" == "${source}" ]]; then
        dir="."
    fi
    (cd "${dir}" >/dev/null 2>&1 && pwd -P)
}
SCRIPT_DIR="$(fct_get_script_dir)"
readonly SCRIPT_DIR

readonly PROJECT_ROOT="${SCRIPT_DIR}"
readonly DEFAULT_SAVE_DIR="${PROJECT_ROOT}/saved_models"
readonly DEFAULT_DATASETS="/home/dhm/dataset/"

# ==============================================================================
# Runtime options
# ==============================================================================

CHECKPOINT=""
DATASETS="${DEFAULT_DATASETS}"
OUTPUT_DIR=""
MODEL_NAME="BASE_Transformer"
DEVICE=""
EMBED_DIM=256
IMG_SIZE=256
BATCH_SIZE=8
N_SAMPLES=6
SEED=42
MAX_TEST_SAMPLES=0
EVAL_BATCHES=0
VERBOSE=0
NO_COLOR="${NO_COLOR:-}"

# ==============================================================================
# Internal state
# ==============================================================================

TMP_DIR=""
POSITIONAL_ARGS=()

# ==============================================================================
# Execution mode helpers
# ==============================================================================

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    IS_SOURCED=1
fi
readonly IS_SOURCED

fct_exit() {
    local code="${1:-0}"
    if [[ "${IS_SOURCED}" -eq 1 ]]; then
        return "${code}"
    fi
    exit "${code}"
}

# ==============================================================================
# Strict mode
# ==============================================================================

fct_enable_strict_mode() {
    set -euo pipefail
    set -o errtrace
}

# ==============================================================================
# Logging
# ==============================================================================

fct_timestamp() { date '+%Y-%m-%d %H:%M:%S%z'; }

fct_ansi() {
    local code="${1}"
    if [[ -t 2 && -z "${NO_COLOR}" ]]; then
        printf '\033[%sm' "${code}"
    fi
}

fct_log() {
    local level="${1}"
    shift
    local message="$*"
    local ts
    ts="$(fct_timestamp)"
    local plain="${ts} [${SCRIPT_NAME}] ${level}: ${message}"
    local rendered="${plain}"

    if [[ -t 2 && -z "${NO_COLOR}" ]]; then
        local prefix=""
        local reset=""
        reset="$(fct_ansi '0')"
        case "${level}" in
        DEBUG) prefix="$(fct_ansi '36')" ;;
        INFO) prefix="$(fct_ansi '32')" ;;
        WARN) prefix="$(fct_ansi '33')" ;;
        ERROR) prefix="$(fct_ansi '31')" ;;
        *) prefix="" ;;
        esac
        rendered="${prefix}${plain}${reset}"
    fi

    printf '%s\n' "${rendered}" >&2
}

log_debug() { if [[ "${VERBOSE}" -eq 1 ]]; then fct_log "DEBUG" "$@"; fi; }
log_info() { fct_log "INFO" "$@"; }
log_warn() { fct_log "WARN" "$@"; }
log_error() { fct_log "ERROR" "$@"; }

die() {
    local message="${1:-Unknown error}"
    local exit_code="${2:-1}"
    log_error "${message}"
    fct_exit "${exit_code}"
}

# ==============================================================================
# Usage
# ==============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}
${SCRIPT_DESCRIPTION}

Usage:
  ${SCRIPT_NAME} [options]

Options:
  -h, --help              Show this help and exit
  -V, --version           Show version and exit
  -v, --verbose           Enable debug logging
      --no-color          Disable colored output
  -c, --checkpoint PATH   Model checkpoint path (auto-detect latest if omitted)
  -d, --datasets PATH     Dataset root directory (default: ${DEFAULT_DATASETS})
  -o, --output-dir PATH   Output directory for all visualizations
  -m, --model NAME        Model architecture name (default: BASE_Transformer)
      --device DEVICE     Device string, e.g. cuda:0 or cpu (auto-detect if omitted)
      --embed-dim N       Embedding dimension (default: 256)
      --img-size N        Input image size (default: 256)
      --batch-size N      Batch size for evaluation (default: 8)
      --n-samples N       Number of sample images to visualize (default: 6)
      --seed N            Random seed (default: 42)
      --max-test-samples N   Max test samples, 0=all (default: 0)
      --eval-batches N    Max eval batches, 0=all (default: 0)

Examples:
  # Auto-find latest checkpoint, generate all visualizations
  ${SCRIPT_NAME}

  # Specify a specific checkpoint
  ${SCRIPT_NAME} -c saved_models/fed_train_20260505_231548/model_best.pth

  # Use ChangeFormerV6 model, output to custom dir
  ${SCRIPT_NAME} -m ChangeFormerV6 -o ./my_viz_results

  # Quick sanity check with fewer samples
  ${SCRIPT_NAME} --max-test-samples 200 --n-samples 3 --eval-batches 5
EOF
}

show_version() {
    printf '%s\n' "${SCRIPT_NAME} v${SCRIPT_VERSION}"
}

# ==============================================================================
# Argument parsing
# ==============================================================================

fct_parse_arguments() {
    POSITIONAL_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
        -h | --help)
            usage
            fct_exit 0
            ;;
        -V | --version)
            show_version
            fct_exit 0
            ;;
        -v | --verbose)
            VERBOSE=1
            shift
            ;;
        --no-color)
            NO_COLOR="1"
            shift
            ;;
        -c | --checkpoint)
            if [[ $# -lt 2 ]]; then die "Option --checkpoint requires a path." 2; fi
            CHECKPOINT="${2}"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#*=}"
            shift
            ;;
        -d | --datasets)
            if [[ $# -lt 2 ]]; then die "Option --datasets requires a path." 2; fi
            DATASETS="${2}"
            shift 2
            ;;
        --datasets=*)
            DATASETS="${1#*=}"
            shift
            ;;
        -o | --output-dir)
            if [[ $# -lt 2 ]]; then die "Option --output-dir requires a path." 2; fi
            OUTPUT_DIR="${2}"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        -m | --model)
            if [[ $# -lt 2 ]]; then die "Option --model requires a name." 2; fi
            MODEL_NAME="${2}"
            shift 2
            ;;
        --model=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        --device)
            if [[ $# -lt 2 ]]; then die "Option --device requires a value." 2; fi
            DEVICE="${2}"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --embed-dim)
            if [[ $# -lt 2 ]]; then die "Option --embed-dim requires a number." 2; fi
            EMBED_DIM="${2}"
            shift 2
            ;;
        --img-size)
            if [[ $# -lt 2 ]]; then die "Option --img-size requires a number." 2; fi
            IMG_SIZE="${2}"
            shift 2
            ;;
        --batch-size)
            if [[ $# -lt 2 ]]; then die "Option --batch-size requires a number." 2; fi
            BATCH_SIZE="${2}"
            shift 2
            ;;
        --n-samples)
            if [[ $# -lt 2 ]]; then die "Option --n-samples requires a number." 2; fi
            N_SAMPLES="${2}"
            shift 2
            ;;
        --seed)
            if [[ $# -lt 2 ]]; then die "Option --seed requires a number." 2; fi
            SEED="${2}"
            shift 2
            ;;
        --max-test-samples)
            if [[ $# -lt 2 ]]; then die "Option --max-test-samples requires a number." 2; fi
            MAX_TEST_SAMPLES="${2}"
            shift 2
            ;;
        --eval-batches)
            if [[ $# -lt 2 ]]; then die "Option --eval-batches requires a number." 2; fi
            EVAL_BATCHES="${2}"
            shift 2
            ;;
        --)
            shift
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        -*)
            die "Unknown option: $1" 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
        esac
    done
}

# ==============================================================================
# Cleanup & traps
# ==============================================================================

cleanup() {
    local exit_status=$?
    set +e
    if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}" 2>/dev/null || true
    fi
    return "${exit_status}"
}

fct_on_error() {
    local exit_status=$?
    local line_no="${1:-?}"
    local command="${2:-?}"
    trap - ERR
    log_error "Command failed (exit ${exit_status}) at line ${line_no}: ${command}"
    exit "${exit_status}"
}

fct_on_signal() {
    local signal="${1:-INT}"
    local exit_code=130
    case "${signal}" in
    INT) exit_code=130 ;;
    TERM) exit_code=143 ;;
    *) exit_code=1 ;;
    esac
    log_warn "Received ${signal}, exiting."
    exit "${exit_code}"
}

fct_setup_traps() {
    trap 'cleanup' EXIT
    trap 'fct_on_error "${LINENO}" "${BASH_COMMAND}"' ERR
    trap 'fct_on_signal INT' INT
    trap 'fct_on_signal TERM' TERM
}

# ==============================================================================
# Checkpoint discovery
# ==============================================================================

fct_find_latest_checkpoint() {
    local save_dir="${1}"
    local best_model=""

    if [[ ! -d "${save_dir}" ]]; then
        die "Save directory not found: ${save_dir}" 1
    fi

    local latest_run=""
    for run_dir in "${save_dir}"/fed_train_*; do
        if [[ -d "${run_dir}" ]]; then
            latest_run="${run_dir}"
        fi
    done

    if [[ -z "${latest_run}" ]]; then
        die "No training run directories found in ${save_dir}" 1
    fi

    best_model="${latest_run}/model_best.pth"
    if [[ ! -f "${best_model}" ]]; then
        die "model_best.pth not found in ${latest_run}" 1
    fi

    printf '%s' "${best_model}"
}

fct_find_log_for_checkpoint() {
    local checkpoint_dir
    checkpoint_dir="$(dirname "${1}")"
    local run_dir_name
    run_dir_name="$(basename "${checkpoint_dir}")"
    local ts_part="${run_dir_name#fed_train_}"

    local log_dir="${PROJECT_ROOT}/logs"
    if [[ ! -d "${log_dir}" ]]; then
        return 1
    fi

    local latest_log=""
    for log_file in "${log_dir}"/*.log; do
        if [[ -f "${log_file}" ]]; then
            latest_log="${log_file}"
        fi
    done

    if [[ -n "${latest_log}" ]]; then
        printf '%s' "${latest_log}"
        return 0
    fi
    return 1
}

# ==============================================================================
# Device detection
# ==============================================================================

fct_detect_device() {
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        printf 'cuda:0'
    else
        printf 'cpu'
    fi
}

# ==============================================================================
# Training curves from log
# ==============================================================================

fct_generate_training_curves() {
    local log_file="${1}"
    local out_dir="${2}"

    if [[ ! -f "${log_file}" ]]; then
        log_warn "Log file not found: ${log_file}, skipping training curves"
        return 0
    fi

    log_info "Parsing training log for metric curves..."

    local curves_script
    curves_script="$(cat <<'PYEOF'
import sys
import re
import json

log_file = sys.argv[1]
out_json = sys.argv[2]

metrics_history = {}
current_round = None

round_re = re.compile(r"训练轮次:\s+(\d+)/\d+")
metric_res = [
    (r"acc:\s+([\d.]+)", "acc"),
    (r"miou:\s+([\d.]+)", "miou"),
    (r"mf1:\s+([\d.]+)", "mf1"),
]
per_class_res = [
    (r"iou_0:\s+([\d.]+)", "iou_0"),
    (r"iou_1:\s+([\d.]+)", "iou_1"),
    (r"F1_0:\s+([\d.]+)", "F1_0"),
    (r"F1_1:\s+([\d.]+)", "F1_1"),
    (r"recall_0:\s+([\d.]+)", "recall_0"),
    (r"recall_1:\s+([\d.]+)", "recall_1"),
    (r"precision_0:\s+([\d.]+)", "precision_0"),
    (r"precision_1:\s+([\d.]+)", "precision_1"),
]

round_has_metrics = False

with open(log_file, "r") as f:
    for line in f:
        m = round_re.search(line)
        if m:
            if round_has_metrics and current_round is not None:
                pass
            current_round = int(m.group(1))
            round_has_metrics = False
            continue

        if current_round is not None:
            for pattern, key in metric_res:
                pm = re.search(pattern, line)
                if pm:
                    if key not in metrics_history:
                        metrics_history[key] = []
                    metrics_history[key].append(float(pm.group(1)))
                    round_has_metrics = True

            for pattern, key in per_class_res:
                pm = re.search(pattern, line)
                if pm:
                    if key not in metrics_history:
                        metrics_history[key] = []
                    metrics_history[key].append(float(pm.group(1)))
                    round_has_metrics = True

if metrics_history:
    lengths = [len(v) for v in metrics_history.values()]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        metrics_history = {k: v[:min_len] for k, v in metrics_history.items()}

    with open(out_json, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Parsed {len(next(iter(metrics_history.values())))} rounds of metrics")
else:
    print("No metrics found in log")
    sys.exit(1)
PYEOF
)"

    local metrics_json="${out_dir}/metrics_history.json"
    python3 -c "${curves_script}" "${log_file}" "${metrics_json}" || {
        log_warn "Failed to parse training log, skipping training curves"
        return 0
    }

    log_info "Generating training curve plots..."
    local plot_script
    plot_script="$(cat <<'PYEOF'
import sys
import json
sys.path.insert(0, sys.argv[3])
from visualization.training_curves import plot_training_curves

with open(sys.argv[1]) as f:
    metrics = json.load(f)
plot_training_curves(metrics, save_path=sys.argv[2], title="Federated Training Metrics")
print(f"Training curves saved to {sys.argv[2]}")
PYEOF
)"
    python3 -c "${plot_script}" "${metrics_json}" "${out_dir}/training_curves.png" "${PROJECT_ROOT}" || {
        log_warn "Failed to generate training curves"
        return 0
    }

    log_info "Training curves saved to ${out_dir}/training_curves.png"
}

# ==============================================================================
# Summary report
# ==============================================================================

fct_generate_summary() {
    local out_dir="${1}"
    local checkpoint="${2}"
    local log_file="${3:-}"

    local summary="${out_dir}/summary.txt"

    {
        printf '=%.0s' {1..70}
        printf '\n'
        printf "FederatedRSCD Visualization Summary\n"
        printf "Generated: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
        printf '=%.0s' {1..70}
        printf '\n\n'

        printf "Checkpoint: %s\n" "${checkpoint}"
        printf "Model:      %s\n" "${MODEL_NAME}"
        printf "Datasets:   %s\n" "${DATASETS}"
        printf "Device:     %s\n" "${DEVICE}"
        printf '\n'

        printf "Generated files:\n"
        local f
        for f in "${out_dir}"/*; do
            if [[ -f "${f}" ]]; then
                local fsize
                fsize="$(du -h "${f}" | cut -f1)"
                printf "  %-40s %s\n" "$(basename "${f}")" "${fsize}"
            fi
        done
        printf '\n'

        if [[ -f "${out_dir}/metrics_history.json" ]]; then
            printf "Last recorded metrics:\n"
            python3 -c "
import json
with open('${out_dir}/metrics_history.json') as f:
    m = json.load(f)
if m:
    n = len(next(iter(m.values())))
    for k in ['acc','miou','mf1','iou_0','iou_1','F1_0','F1_1']:
        if k in m:
            print(f'  {k}: {m[k][-1]:.4f}')
" 2>/dev/null || true
            printf '\n'
        fi
    } > "${summary}"

    log_info "Summary report saved to ${summary}"
}

# ==============================================================================
# Main logic
# ==============================================================================

fct_execute_this() {
    if [[ -z "${DEVICE}" ]]; then
        DEVICE="$(fct_detect_device)"
        log_info "Auto-detected device: ${DEVICE}"
    fi

    if [[ -z "${CHECKPOINT}" ]]; then
        log_info "No checkpoint specified, searching in ${DEFAULT_SAVE_DIR}..."
        CHECKPOINT="$(fct_find_latest_checkpoint "${DEFAULT_SAVE_DIR}")"
        log_info "Found checkpoint: ${CHECKPOINT}"
    fi

    if [[ ! -f "${CHECKPOINT}" ]]; then
        die "Checkpoint not found: ${CHECKPOINT}" 1
    fi

    local checkpoint_dir
    checkpoint_dir="$(dirname "${CHECKPOINT}")"

    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="${checkpoint_dir}/viz"
    fi
    mkdir -p "${OUTPUT_DIR}"

    log_info "Output directory: ${OUTPUT_DIR}"
    log_info "Checkpoint: ${CHECKPOINT}"
    log_info "Model: ${MODEL_NAME}"
    log_info "Datasets: ${DATASETS}"

    log_info "Running model evaluation and visualization..."
    PYTHONPATH="${PROJECT_ROOT}" python3 "${PROJECT_ROOT}/tools/visualize_results.py" \
        --checkpoint "${CHECKPOINT}" \
        --model_name "${MODEL_NAME}" \
        --datasets "${DATASETS}" \
        --device "${DEVICE}" \
        --embed_dim "${EMBED_DIM}" \
        --img_size "${IMG_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --n_samples "${N_SAMPLES}" \
        --seed "${SEED}" \
        --max_test_samples "${MAX_TEST_SAMPLES}" \
        --eval_batches "${EVAL_BATCHES}" \
        --output_dir "${OUTPUT_DIR}"

    log_info "Visualization tool completed (7 plots generated)"

    local log_file
    if log_file="$(fct_find_log_for_checkpoint "${CHECKPOINT}")"; then
        log_info "Found training log: ${log_file}"
        fct_generate_training_curves "${log_file}" "${OUTPUT_DIR}"
    else
        log_warn "No training log found, skipping training curves"
    fi

    fct_generate_summary "${OUTPUT_DIR}" "${CHECKPOINT}" "${log_file:-}"

    printf '\n'
    log_info "All done! Results saved to: ${OUTPUT_DIR}"
    printf '=%.0s' {1..60}
    printf '\n'
    printf "Output files:\n"
    local f
    for f in "${OUTPUT_DIR}"/*; do
        if [[ -f "${f}" ]]; then
            local fsize
            fsize="$(du -h "${f}" | cut -f1)"
            printf "  %s  %s\n" "$(basename "${f}")" "${fsize}"
        fi
    done
}

main() {
    fct_enable_strict_mode
    fct_setup_traps

    fct_parse_arguments "$@"

    TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${SCRIPT_NAME}.XXXXXXXX")" || die "Failed to create temp dir." 1

    log_debug "Script dir: ${SCRIPT_DIR}"
    log_debug "Temp dir: ${TMP_DIR}"

    fct_execute_this
}

if [[ "${IS_SOURCED}" -eq 0 ]]; then
    main "$@"
fi
