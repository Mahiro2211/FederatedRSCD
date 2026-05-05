#!/usr/bin/env bash
#
# ==============================================================================
# Post-training Visualization Pipeline (Multi-model)
# ==============================================================================
# Auto-discover trained model checkpoints from a session directory and generate
# full visualizations for each model.
#
# Expected checkpoint naming convention (produced by train_all.sh):
#   <session_dir>/SiamUnet_diff_best.pth
#   <session_dir>/BASE_Transformer_best.pth
#   <session_dir>/ChangeFormerV6_best.pth
#
# The model architecture name is extracted from the filename automatically.
# ==============================================================================

readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_DESCRIPTION="Multi-model visualization pipeline for FederatedRSCD"

readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly SCRIPT_NAME="${SCRIPT_PATH##*/}"

fct_get_script_dir() {
    local source="${SCRIPT_PATH}"
    local dir="${source%/*}"
    if [[ "${dir}" == "${source}" ]]; then dir="."; fi
    (cd "${dir}" >/dev/null 2>&1 && pwd -P)
}
SCRIPT_DIR="$(fct_get_script_dir)"
readonly SCRIPT_DIR
readonly PROJECT_ROOT="${SCRIPT_DIR}"

readonly DEFAULT_DATASETS="/home/dhm/dataset/"
readonly DEFAULT_SAVE_ROOT="${PROJECT_ROOT}/saved_models"

# ==============================================================================
# Runtime options
# ==============================================================================

SESSION_DIR=""
DATASETS="${DEFAULT_DATASETS}"
OUTPUT_BASE=""
DEVICE=""
EMBED_DIM=256
IMG_SIZE=256
BATCH_SIZE=8
N_SAMPLES=6
SEED=42
MAX_TEST_SAMPLES=0
EVAL_BATCHES=0
ONLY_MODELS=()
VERBOSE=0
NO_COLOR="${NO_COLOR:-}"

# ==============================================================================
# Internal state
# ==============================================================================

TMP_DIR=""
POSITIONAL_ARGS=()
VIZ_STATS=()

# ==============================================================================
# Execution mode helpers
# ==============================================================================

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then IS_SOURCED=1; fi
readonly IS_SOURCED

fct_exit() {
    local code="${1:-0}"
    if [[ "${IS_SOURCED}" -eq 1 ]]; then return "${code}"; fi
    exit "${code}"
}

# ==============================================================================
# Strict mode
# ==============================================================================

fct_enable_strict_mode() { set -euo pipefail; set -o errtrace; }

# ==============================================================================
# Logging
# ==============================================================================

fct_timestamp() { date '+%Y-%m-%d %H:%M:%S%z'; }

fct_ansi() {
    local code="${1}"
    if [[ -t 2 && -z "${NO_COLOR}" ]]; then printf '\033[%sm' "${code}"; fi
}

fct_log() {
    local level="${1}"; shift
    local message="$*"
    local ts
    ts="$(fct_timestamp)"
    local plain="${ts} [${SCRIPT_NAME}] ${level}: ${message}"
    local rendered="${plain}"

    if [[ -t 2 && -z "${NO_COLOR}" ]]; then
        local prefix="" reset=""
        reset="$(fct_ansi '0')"
        case "${level}" in
        DEBUG) prefix="$(fct_ansi '36')" ;;
        INFO)  prefix="$(fct_ansi '32')" ;;
        WARN)  prefix="$(fct_ansi '33')" ;;
        ERROR) prefix="$(fct_ansi '31')" ;;
        esac
        rendered="${prefix}${plain}${reset}"
    fi
    printf '%s\n' "${rendered}" >&2
}

log_debug() { if [[ "${VERBOSE}" -eq 1 ]]; then fct_log "DEBUG" "$@"; fi; }
log_info()  { fct_log "INFO" "$@"; }
log_warn()  { fct_log "WARN" "$@"; }
log_error() { fct_log "ERROR" "$@"; }

die() {
    local message="${1:-Unknown error}" exit_code="${2:-1}"
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
  ${SCRIPT_NAME} -s <session_dir> [options]
  ${SCRIPT_NAME} [options]  (auto-find latest session)

Options:
  -h, --help                  Show this help and exit
  -V, --version               Show version and exit
  -v, --verbose               Enable debug logging
      --no-color              Disable colored output
  -s, --session-dir PATH      Session directory containing *_best.pth files
                              (auto-find latest under saved_models/ if omitted)
  -d, --datasets PATH         Dataset root directory (default: ${DEFAULT_DATASETS})
  -o, --output-base PATH      Base output directory (default: <session_dir>/viz/)
      --device DEVICE         Device, e.g. cuda:0 or cpu (auto-detect if omitted)
      --embed-dim N           Embedding dimension (default: 256)
      --img-size N            Input image size (default: 256)
      --batch-size N          Evaluation batch size (default: 8)
      --n-samples N           Sample images per model (default: 6)
      --seed N                Random seed (default: 42)
      --max-test-samples N    Max test samples, 0=all (default: 0)
      --eval-batches N        Max eval batches, 0=all (default: 0)
      --only-model MODEL      Visualize ONLY this model (repeatable)

Checkpoint naming convention:
  The script scans for files matching *_best.pth in the session directory.
  The model name is extracted by stripping the "_best.pth" suffix.

  Example:
    SiamUnet_diff_best.pth    -> model_name = SiamUnet_diff
    BASE_Transformer_best.pth -> model_name = BASE_Transformer
    ChangeFormerV6_best.pth   -> model_name = ChangeFormerV6

Examples:
  # Auto-find latest session, visualize all models
  ${SCRIPT_NAME}

  # Visualize a specific session
  ${SCRIPT_NAME} -s saved_models/batch_20260506_120000

  # Visualize only BIT model from a session
  ${SCRIPT_NAME} -s saved_models/batch_20260506_120000 --only-model BASE_Transformer

  # Quick check with fewer samples
  ${SCRIPT_NAME} --max-test-samples 200 --n-samples 3 --eval-batches 5
EOF
}

show_version() { printf '%s\n' "${SCRIPT_NAME} v${SCRIPT_VERSION}"; }

# ==============================================================================
# Argument parsing
# ==============================================================================

fct_parse_arguments() {
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
        -h|--help)       usage; fct_exit 0 ;;
        -V|--version)    show_version; fct_exit 0 ;;
        -v|--verbose)    VERBOSE=1; shift ;;
        --no-color)      NO_COLOR="1"; shift ;;
        -s|--session-dir)
            [[ $# -lt 2 ]] && die "--session-dir requires a path." 2
            SESSION_DIR="${2}"; shift 2 ;;
        --session-dir=*) SESSION_DIR="${1#*=}"; shift ;;
        -d|--datasets)
            [[ $# -lt 2 ]] && die "--datasets requires a path." 2
            DATASETS="${2}"; shift 2 ;;
        --datasets=*)    DATASETS="${1#*=}"; shift ;;
        -o|--output-base)
            [[ $# -lt 2 ]] && die "--output-base requires a path." 2
            OUTPUT_BASE="${2}"; shift 2 ;;
        --output-base=*) OUTPUT_BASE="${1#*=}"; shift ;;
        --device)
            [[ $# -lt 2 ]] && die "--device requires a value." 2
            DEVICE="${2}"; shift 2 ;;
        --device=*)      DEVICE="${1#*=}"; shift ;;
        --embed-dim)
            [[ $# -lt 2 ]] && die "--embed-dim requires a number." 2
            EMBED_DIM="${2}"; shift 2 ;;
        --img-size)
            [[ $# -lt 2 ]] && die "--img-size requires a number." 2
            IMG_SIZE="${2}"; shift 2 ;;
        --batch-size)
            [[ $# -lt 2 ]] && die "--batch-size requires a number." 2
            BATCH_SIZE="${2}"; shift 2 ;;
        --n-samples)
            [[ $# -lt 2 ]] && die "--n-samples requires a number." 2
            N_SAMPLES="${2}"; shift 2 ;;
        --seed)
            [[ $# -lt 2 ]] && die "--seed requires a number." 2
            SEED="${2}"; shift 2 ;;
        --max-test-samples)
            [[ $# -lt 2 ]] && die "--max-test-samples requires a number." 2
            MAX_TEST_SAMPLES="${2}"; shift 2 ;;
        --eval-batches)
            [[ $# -lt 2 ]] && die "--eval-batches requires a number." 2
            EVAL_BATCHES="${2}"; shift 2 ;;
        --only-model)
            [[ $# -lt 2 ]] && die "--only-model requires a model name." 2
            ONLY_MODELS+=("${2}"); shift 2 ;;
        --) shift; POSITIONAL_ARGS+=("$@"); break ;;
        -*) die "Unknown option: $1" 2 ;;
        *)  POSITIONAL_ARGS+=("$1"); shift ;;
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
    local exit_status=$? line_no="${1:-?}" command="${2:-?}"
    trap - ERR
    log_error "Command failed (exit ${exit_status}) at line ${line_no}: ${command}"
    exit "${exit_status}"
}

fct_on_signal() {
    local signal="${1:-INT}" exit_code=130
    case "${signal}" in
    INT) exit_code=130 ;; TERM) exit_code=143 ;; *) exit_code=1 ;;
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
# Session discovery
# ==============================================================================

fct_find_latest_session() {
    local save_root="${1}"
    if [[ ! -d "${save_root}" ]]; then
        die "Save root not found: ${save_root}" 1
    fi

    local latest=""
    local d
    for d in "${save_root}"/batch_*/; do
        if [[ -d "${d}" ]]; then latest="${d}"; fi
    done

    if [[ -z "${latest}" ]]; then
        die "No batch_* session directories found in ${save_root}" 1
    fi

    printf '%s' "${latest%/}"
}

# ==============================================================================
# Checkpoint discovery
# ==============================================================================

fct_discover_checkpoints() {
    local session_dir="${1}"
    local -a checkpoints=()

    local f
    for f in "${session_dir}"/*_best.pth; do
        [[ -f "${f}" ]] || continue

        local basename
        basename="$(basename "${f}")"
        local model_name="${basename%_best.pth}"

        if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
            local match=0 o
            for o in "${ONLY_MODELS[@]}"; do
                if [[ "${model_name}" == "${o}" ]]; then match=1; break; fi
            done
            [[ "${match}" -eq 0 ]] && continue
        fi

        checkpoints+=("${model_name}:${f}")
    done

    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        die "No *_best.pth checkpoints found in ${session_dir}" 1
    fi

    printf '%s\n' "${checkpoints[@]}"
}

# ==============================================================================
# Single model visualization
# ==============================================================================

fct_visualize_single_model() {
    local model_name="${1}"
    local checkpoint="${2}"
    local viz_dir="${3}"

    local model_start_time
    model_start_time="$(date +%s)"

    log_info "Visualizing: ${model_name}"
    log_info "  Checkpoint: ${checkpoint}"
    log_info "  Output:     ${viz_dir}"

    mkdir -p "${viz_dir}"

    PYTHONPATH="${PROJECT_ROOT}" python3 "${PROJECT_ROOT}/tools/visualize_results.py" \
        --checkpoint "${checkpoint}" \
        --model_name "${model_name}" \
        --datasets "${DATASETS}" \
        --device "${DEVICE}" \
        --embed_dim "${EMBED_DIM}" \
        --img_size "${IMG_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --n_samples "${N_SAMPLES}" \
        --seed "${SEED}" \
        --max_test_samples "${MAX_TEST_SAMPLES}" \
        --eval_batches "${EVAL_BATCHES}" \
        --output_dir "${viz_dir}"

    local model_end_time
    model_end_time="$(date +%s)"
    local elapsed=$(( model_end_time - model_start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    local n_files=0
    local ff
    for ff in "${viz_dir}"/*; do
        [[ -f "${ff}" ]] && n_files=$(( n_files + 1 ))
    done

    log_info "${model_name} visualization completed (${n_files} files, ${mins}m ${secs}s)"
    VIZ_STATS+=("${model_name}: ${n_files} files, ${mins}m ${secs}s")
}

# ==============================================================================
# Training curves from log (optional)
# ==============================================================================

fct_find_log_for_session() {
    local log_dir="${PROJECT_ROOT}/logs"
    [[ -d "${log_dir}" ]] || return 1

    local latest=""
    local f
    for f in "${log_dir}"/*.log; do
        [[ -f "${f}" ]] && latest="${f}"
    done

    [[ -n "${latest}" ]] && printf '%s' "${latest}"
}

fct_generate_training_curves() {
    local log_file="${1}"
    local out_dir="${2}"

    [[ -f "${log_file}" ]] || { log_warn "Log not found: ${log_file}"; return 0; }

    log_info "Parsing training log for metric curves..."

    local parse_script
    parse_script="$(cat <<'PYEOF'
import sys, re, json

log_file, out_json = sys.argv[1], sys.argv[2]
metrics_history = {}
current_round = None

round_re = re.compile(r"训练轮次:\s+(\d+)/\d+")
all_patterns = [
    (r"acc:\s+([\d.]+)", "acc"),
    (r"miou:\s+([\d.]+)", "miou"),
    (r"mf1:\s+([\d.]+)", "mf1"),
    (r"iou_0:\s+([\d.]+)", "iou_0"),
    (r"iou_1:\s+([\d.]+)", "iou_1"),
    (r"F1_0:\s+([\d.]+)", "F1_0"),
    (r"F1_1:\s+([\d.]+)", "F1_1"),
    (r"recall_0:\s+([\d.]+)", "recall_0"),
    (r"recall_1:\s+([\d.]+)", "recall_1"),
    (r"precision_0:\s+([\d.]+)", "precision_0"),
    (r"precision_1:\s+([\d.]+)", "precision_1"),
]

with open(log_file) as f:
    for line in f:
        m = round_re.search(line)
        if m:
            current_round = int(m.group(1))
            continue
        if current_round is not None:
            for pat, key in all_patterns:
                pm = re.search(pat, line)
                if pm:
                    metrics_history.setdefault(key, []).append(float(pm.group(1)))

if not metrics_history:
    sys.exit(1)

lengths = [len(v) for v in metrics_history.values()]
if len(set(lengths)) > 1:
    ml = min(lengths)
    metrics_history = {k: v[:ml] for k, v in metrics_history.items()}

with open(out_json, "w") as f:
    json.dump(metrics_history, f, indent=2)
print(f"Parsed {len(next(iter(metrics_history.values())))} rounds")
PYEOF
)"

    local metrics_json="${out_dir}/metrics_history.json"
    python3 -c "${parse_script}" "${log_file}" "${metrics_json}" || {
        log_warn "Failed to parse log, skipping training curves"
        return 0
    }

    local plot_script
    plot_script="$(cat <<'PYEOF'
import sys, json
sys.path.insert(0, sys.argv[3])
from visualization.training_curves import plot_training_curves

with open(sys.argv[1]) as f:
    metrics = json.load(f)
plot_training_curves(metrics, save_path=sys.argv[2], title="Federated Training Metrics")
PYEOF
)"
    python3 -c "${plot_script}" "${metrics_json}" "${out_dir}/training_curves.png" "${PROJECT_ROOT}" || {
        log_warn "Failed to generate training curves"
        return 0
    }

    log_info "Training curves saved to ${out_dir}/training_curves.png"
}

# ==============================================================================
# Summary
# ==============================================================================

fct_print_summary() {
    local session_dir="${1}"
    local viz_base="${2}"

    printf '\n'
    printf '%s\n' "============================================================"
    printf '%s\n' "  Visualization Summary"
    printf '%s\n' "  Session: $(basename "${session_dir}")"
    printf '%s\n' "  Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s\n' "============================================================"
    printf '\n'

    local stat
    for stat in "${VIZ_STATS[@]+"${VIZ_STATS[@]}"}"; do
        printf "  %s\n" "${stat}"
    done
    printf '\n'

    printf "%s\n" "Output structure:"
    local model_dir
    for model_dir in "${viz_base}"/*/; do
        [[ -d "${model_dir}" ]] || continue
        printf "  %s/\n" "$(basename "${model_dir}")"
        local f
        for f in "${model_dir}"*; do
            [[ -f "${f}" ]] || continue
            local fsize
            fsize="$(du -h "${f}" | cut -f1)"
            printf "    %-45s %s\n" "$(basename "${f}")" "${fsize}"
        done
    done
    printf '\n'
}

# ==============================================================================
# Main logic
# ==============================================================================

fct_execute_this() {
    if [[ -z "${DEVICE}" ]]; then
        DEVICE="$(fct_detect_device)"
        log_info "Auto-detected device: ${DEVICE}"
    fi

    if [[ -z "${SESSION_DIR}" ]]; then
        log_info "No session specified, searching in ${DEFAULT_SAVE_ROOT}..."
        SESSION_DIR="$(fct_find_latest_session "${DEFAULT_SAVE_ROOT}")"
        log_info "Found session: ${SESSION_DIR}"
    fi

    if [[ ! -d "${SESSION_DIR}" ]]; then
        die "Session directory not found: ${SESSION_DIR}" 1
    fi

    if [[ -z "${OUTPUT_BASE}" ]]; then
        OUTPUT_BASE="${SESSION_DIR}/viz"
    fi
    mkdir -p "${OUTPUT_BASE}"

    local -a checkpoint_entries
    mapfile -t checkpoint_entries < <(fct_discover_checkpoints "${SESSION_DIR}")

    log_info "Session:     ${SESSION_DIR}"
    log_info "Output base: ${OUTPUT_BASE}"
    log_info "Found ${#checkpoint_entries[@]} model checkpoint(s)"

    local entry model_name checkpoint viz_dir
    for entry in "${checkpoint_entries[@]}"; do
        model_name="${entry%%:*}"
        checkpoint="${entry#*:}"
        viz_dir="${OUTPUT_BASE}/${model_name}"

        log_info "----------------------------------------"
        fct_visualize_single_model "${model_name}" "${checkpoint}" "${viz_dir}"
    done

    local log_file
    if log_file="$(fct_find_log_for_session)"; then
        log_info "Found training log: ${log_file}"
        fct_generate_training_curves "${log_file}" "${OUTPUT_BASE}"
    else
        log_warn "No training log found, skipping training curves"
    fi

    fct_print_summary "${SESSION_DIR}" "${OUTPUT_BASE}"
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

if [[ "${IS_SOURCED}" -eq 0 ]]; then main "$@"; fi
