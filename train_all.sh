#!/usr/bin/env bash
#
# ==============================================================================
# Batch Training Script for FederatedRSCD
# ==============================================================================
# Train multiple model architectures sequentially with clear weight naming.
#
# Supported models:
#   SiamUnet_diff    - Siamese U-Net (difference)
#   BASE_Transformer - BIT (ResNet18 + Transformer)
#   ChangeFormerV6   - ChangeFormer V6
#
# Weights are saved as:
#   <session_dir>/SiamUnet_diff_best.pth
#   <session_dir>/BASE_Transformer_best.pth
#   <session_dir>/ChangeFormerV6_best.pth
# ==============================================================================

readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DESCRIPTION="Batch training pipeline for FederatedRSCD (SiamUnet_diff + BIT + ChangeFormerV6)"

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
readonly MODELS=("SiamUnet_diff" "BASE_Transformer" "ChangeFormerV6")

# ==============================================================================
# Runtime options
# ==============================================================================

DATASETS="${DEFAULT_DATASETS}"
SAVE_ROOT="${DEFAULT_SAVE_ROOT}"
SESSION_DIR=""
DEVICE=""
NUM_EPOCHS=50
NUM_CLIENT_EPOCH=2
BATCH_SIZE=16
LR=0.0001
LOSS_TYPE="ce_dice"
FRAC=0.5
EVAL_INTERVAL=1
EMBED_DIM=256
IMG_SIZE=256
SKIP_MODELS=()
ONLY_MODELS=()
VERBOSE=0
NO_COLOR="${NO_COLOR:-}"

# ==============================================================================
# Internal state
# ==============================================================================

TMP_DIR=""
POSITIONAL_ARGS=()
TRAIN_STATS=()

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
  ${SCRIPT_NAME} [options]

Options:
  -h, --help                  Show this help and exit
  -V, --version               Show version and exit
  -v, --verbose               Enable debug logging
      --no-color              Disable colored output
  -d, --datasets PATH         Dataset root directory (default: ${DEFAULT_DATASETS})
  -s, --save-root PATH        Root directory for saving sessions (default: saved_models/)
      --session-dir PATH      Explicit session directory (auto-generated if omitted)
      --device DEVICE         Device string, e.g. cuda:0 or cpu (auto-detect if omitted)
      --num-epochs N          Federated rounds per model (default: 50)
      --num-client-epoch N    Local epochs per client per round (default: 2)
      --batch-size N          Batch size (default: 16)
      --lr RATE               Learning rate (default: 0.0001)
      --loss-type TYPE        Loss function: ce|focal|dice|ce_dice (default: ce_dice)
      --frac RATIO            Client participation fraction (default: 0.5)
      --eval-interval N       Evaluation interval in rounds (default: 1)
      --embed-dim N           Embedding dimension for ChangeFormerV6 (default: 256)
      --img-size N            Input image size (default: 256)
      --skip-model MODEL      Skip a specific model (repeatable)
      --only-model MODEL      Train ONLY this model (repeatable, overrides default set)

Models trained (in order):
  1. SiamUnet_diff    - Siamese U-Net (difference), ~1.4M params
  2. BASE_Transformer - BIT (ResNet18 + Transformer), ~11.9M params
  3. ChangeFormerV6   - ChangeFormer V6 (lightweight encoder), ~41.0M params

Examples:
  # Train all 3 models with default settings
  ${SCRIPT_NAME}

  # Train only BIT and ChangeFormerV6 for 50 rounds
  ${SCRIPT_NAME} --only-model BASE_Transformer --only-model ChangeFormerV6 --num-epochs 50

  # Train all models with focal loss, higher participation
  ${SCRIPT_NAME} --loss-type focal --frac 0.8 --num-epochs 30

  # Resume visualization for an existing session (skip training, see visualize_all.sh)
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
        -d|--datasets)
            [[ $# -lt 2 ]] && die "--datasets requires a path." 2
            DATASETS="${2}"; shift 2 ;;
        --datasets=*)    DATASETS="${1#*=}"; shift ;;
        -s|--save-root)
            [[ $# -lt 2 ]] && die "--save-root requires a path." 2
            SAVE_ROOT="${2}"; shift 2 ;;
        --save-root=*)   SAVE_ROOT="${1#*=}"; shift ;;
        --session-dir)
            [[ $# -lt 2 ]] && die "--session-dir requires a path." 2
            SESSION_DIR="${2}"; shift 2 ;;
        --session-dir=*) SESSION_DIR="${1#*=}"; shift ;;
        --device)
            [[ $# -lt 2 ]] && die "--device requires a value." 2
            DEVICE="${2}"; shift 2 ;;
        --device=*)      DEVICE="${1#*=}"; shift ;;
        --num-epochs)
            [[ $# -lt 2 ]] && die "--num-epochs requires a number." 2
            NUM_EPOCHS="${2}"; shift 2 ;;
        --num-client-epoch)
            [[ $# -lt 2 ]] && die "--num-client-epoch requires a number." 2
            NUM_CLIENT_EPOCH="${2}"; shift 2 ;;
        --batch-size)
            [[ $# -lt 2 ]] && die "--batch-size requires a number." 2
            BATCH_SIZE="${2}"; shift 2 ;;
        --lr)
            [[ $# -lt 2 ]] && die "--lr requires a number." 2
            LR="${2}"; shift 2 ;;
        --loss-type)
            [[ $# -lt 2 ]] && die "--loss-type requires a value." 2
            LOSS_TYPE="${2}"; shift 2 ;;
        --loss-type=*)   LOSS_TYPE="${1#*=}"; shift ;;
        --frac)
            [[ $# -lt 2 ]] && die "--frac requires a number." 2
            FRAC="${2}"; shift 2 ;;
        --eval-interval)
            [[ $# -lt 2 ]] && die "--eval-interval requires a number." 2
            EVAL_INTERVAL="${2}"; shift 2 ;;
        --embed-dim)
            [[ $# -lt 2 ]] && die "--embed-dim requires a number." 2
            EMBED_DIM="${2}"; shift 2 ;;
        --img-size)
            [[ $# -lt 2 ]] && die "--img-size requires a number." 2
            IMG_SIZE="${2}"; shift 2 ;;
        --skip-model)
            [[ $# -lt 2 ]] && die "--skip-model requires a model name." 2
            SKIP_MODELS+=("${2}"); shift 2 ;;
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
# Model list resolution
# ==============================================================================

fct_resolve_model_list() {
    local -a models=()

    if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
        models=("${ONLY_MODELS[@]}")
    else
        models=("${MODELS[@]}")
    fi

    local -a filtered=()
    local m skip
    for m in "${models[@]}"; do
        skip=0
        local s
        for s in "${SKIP_MODELS[@]+"${SKIP_MODELS[@]}"}"; do
            if [[ "${m}" == "${s}" ]]; then skip=1; break; fi
        done
        if [[ "${skip}" -eq 0 ]]; then
            filtered+=("${m}")
        fi
    done

    printf '%s\n' "${filtered[@]}"
}

# ==============================================================================
# Single model training
# ==============================================================================

fct_train_single_model() {
    local model_name="${1}"
    local session_dir="${2}"
    local model_start_time
    model_start_time="$(date +%s)"

    local weight_name="${model_name}_best.pth"
    local target_path="${session_dir}/${weight_name}"

    log_info "========================================"
    log_info "Training: ${model_name}"
    log_info "Target:   ${target_path}"
    log_info "========================================"

    local tmp_save
    tmp_save="$(mktemp -d "${TMP_DIR}/train_${model_name}_XXXXXX")"

    PYTHONPATH="${PROJECT_ROOT}" python3 "${PROJECT_ROOT}/main.py" \
        --model_name "${model_name}" \
        --datasets "${DATASETS}" \
        --device "${DEVICE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --num_client_epoch "${NUM_CLIENT_EPOCH}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --loss_type "${LOSS_TYPE}" \
        --frac "${FRAC}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --embed_dim "${EMBED_DIM}" \
        --img_size "${IMG_SIZE}" \
        --save_dir "${tmp_save}" \
        --no-visualize

    local best_ckpt=""
    local run_dir
    for run_dir in "${tmp_save}"/fed_train_*/; do
        if [[ -f "${run_dir}model_best.pth" ]]; then
            best_ckpt="${run_dir}model_best.pth"
        fi
    done

    if [[ -z "${best_ckpt}" ]]; then
        log_error "No model_best.pth found for ${model_name} in ${tmp_save}"
        return 1
    fi

    cp "${best_ckpt}" "${target_path}"
    log_info "Copied ${best_ckpt} -> ${target_path}"

    rm -rf "${tmp_save}"

    local model_end_time
    model_end_time="$(date +%s)"
    local elapsed=$(( model_end_time - model_start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    log_info "${model_name} completed in ${mins}m ${secs}s"
    TRAIN_STATS+=("${model_name}: ${mins}m ${secs}s -> ${weight_name}")
}

# ==============================================================================
# Batch summary
# ==============================================================================

fct_print_summary() {
    local session_dir="${1}"

    printf '\n'
    printf '%s\n' "============================================================"
    printf '%s\n' "  Batch Training Summary"
    printf '%s\n' "  Session: $(basename "${session_dir}")"
    printf '%s\n' "  Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s\n' "============================================================"
    printf '\n'

    printf "%s\n" "Configuration:"
    printf "  %-24s %s\n" "Datasets:" "${DATASETS}"
    printf "  %-24s %s\n" "Loss:" "${LOSS_TYPE}"
    printf "  %-24s %s\n" "Epochs:" "${NUM_EPOCHS}"
    printf "  %-24s %s\n" "Client epochs:" "${NUM_CLIENT_EPOCH}"
    printf "  %-24s %s\n" "Batch size:" "${BATCH_SIZE}"
    printf "  %-24s %s\n" "LR:" "${LR}"
    printf "  %-24s %s\n" "Frac:" "${FRAC}"
    printf "  %-24s %s\n" "Device:" "${DEVICE}"
    printf '\n'

    printf "%s\n" "Training results:"
    local stat
    for stat in "${TRAIN_STATS[@]+"${TRAIN_STATS[@]}"}"; do
        printf "  %s\n" "${stat}"
    done
    printf '\n'

    printf "%s\n" "Weight files:"
    local f
    for f in "${session_dir}"/*_best.pth; do
        if [[ -f "${f}" ]]; then
            local fsize
            fsize="$(du -h "${f}" | cut -f1)"
            printf "  %-45s %s\n" "$(basename "${f}")" "${fsize}"
        fi
    done
    printf '\n'

    printf "%s\n" "Next step - visualize all results:"
    printf "  ./visualize_all.sh -s ${session_dir}\n"
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
        SESSION_DIR="${SAVE_ROOT}/batch_$(date '+%Y%m%d_%H%M%S')"
    fi
    mkdir -p "${SESSION_DIR}"

    local -a model_list
    mapfile -t model_list < <(fct_resolve_model_list)

    if [[ ${#model_list[@]} -eq 0 ]]; then
        die "No models to train (all filtered out)." 1
    fi

    log_info "Session directory: ${SESSION_DIR}"
    log_info "Models to train: ${model_list[*]}"
    log_info "Total: ${#model_list[@]} model(s)"

    local batch_start_time
    batch_start_time="$(date +%s)"

    local model
    for model in "${model_list[@]}"; do
        fct_train_single_model "${model}" "${SESSION_DIR}"
    done

    local batch_end_time
    batch_end_time="$(date +%s)"
    local total=$(( batch_end_time - batch_start_time ))
    local total_mins=$(( total / 60 ))
    local total_secs=$(( total % 60 ))

    log_info "All models trained in ${total_mins}m ${total_secs}s"

    fct_print_summary "${SESSION_DIR}"
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
