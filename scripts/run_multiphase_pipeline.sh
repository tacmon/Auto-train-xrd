#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yaml"
SERVICE_NAME="xrd-service"
CONTAINER_ROOT="/workspace/project"
NOVEL_SPACE="$REPO_ROOT/libs/XRD-1.1/Novel-Space"
CONTAINER_NOVEL_SPACE="$CONTAINER_ROOT/libs/XRD-1.1/Novel-Space"
MP_TOOL="$CONTAINER_ROOT/.codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py"

PHASE_LABELS=()
MP_IDS=()
LOCAL_CIFS=()
MANUAL_REFERENCES=()
SPECTRA_SOURCE="$REPO_ROOT/data"
RUN_NAME=""
NUM_SPECTRA="50"
XRD_EPOCHS="50"
PDF_EPOCHS="50"
MIN_ANGLE="20.0"
MAX_ANGLE="60.0"

usage() {
  cat <<'EOF'
Usage:
  run_multiphase_pipeline.sh \
    --phase-label LABEL \
    [--phase-label LABEL ...] \
    [--mp-material-id mp-xxxx ...] \
    [--local-cif path/to/file.cif ...]

Options:
  --phase-label LABEL      Phase label used only for run naming/reporting. Repeatable.
  --mp-material-id ID      Materials Project material ID to download. Repeatable.
  --local-cif PATH         Local CIF to include in this run. Repeatable.
  --manual-reference MAP   Create References entry directly and skip tabulate_cifs.
                           Format: ReferenceName=SourceCifFilename
  --spectra-source PATH    Real spectra source. Default: ./data
  --run-name NAME          Explicit run directory name
  --num-spectra N          Simulated spectra per phase. Default: 50
  --xrd-epochs N           XRD training epochs. Default: 50
  --pdf-epochs N           PDF training epochs. Default: 50
  --min-angle VALUE        Lower two-theta bound. Default: 20.0
  --max-angle VALUE        Upper two-theta bound. Default: 60.0
EOF
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

compose_exec() {
  LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" bash -lc "$1"
}

copy_spectra() {
  local source=$1
  local target=$2
  local copied=0

  rm -rf "$target"
  mkdir -p "$target"

  if [[ -f "$source" ]]; then
    case "${source,,}" in
      *.txt|*.xy|*.gk)
        cp "$source" "$target/"
        copied=1
        ;;
      *)
        echo "[ERROR] Unsupported spectra file: $source" >&2
        exit 1
        ;;
    esac
  elif [[ -d "$source" ]]; then
    while IFS= read -r -d '' file; do
      local relative
      relative=${file#"$source"/}
      mkdir -p "$target/$(dirname "$relative")"
      cp "$file" "$target/$relative"
      copied=$((copied + 1))
    done < <(find "$source" -type f \( -iname '*.txt' -o -iname '*.xy' -o -iname '*.gk' \) -print0)
  else
    echo "[ERROR] Spectra source does not exist: $source" >&2
    exit 1
  fi

  if [[ "$copied" -eq 0 ]]; then
    echo "[ERROR] No supported spectra files (*.txt, *.xy, *.gk) were found in: $source" >&2
    exit 1
  fi
}

link_workspace() {
  mkdir -p "$NOVEL_SPACE/figure"
  rm -rf "$NOVEL_SPACE/Spectra" "$NOVEL_SPACE/All_CIFs" "$NOVEL_SPACE/Models" \
    "$NOVEL_SPACE/References" "$NOVEL_SPACE/figure/real_data"
  ln -snf "soft_link/Spectra/$RUN_NAME" "$NOVEL_SPACE/Spectra"
  ln -snf "soft_link/All_CIFs/$RUN_NAME" "$NOVEL_SPACE/All_CIFs"
  ln -snf "../soft_link/figure/$RUN_NAME" "$NOVEL_SPACE/figure/real_data"
}

persist_dir_as_link() {
  local local_dir=$1
  local target_dir=$2
  local relative_target="soft_link/All_CIFs/$RUN_NAME/$(basename "$target_dir")"
  if [[ -d "$local_dir" && ! -L "$local_dir" ]]; then
    mv "$local_dir" "$target_dir"
    ln -snf "$relative_target" "$local_dir"
  fi
}

require_nonempty_list() {
  local label=$1
  local count=$2
  if [[ "$count" -eq 0 ]]; then
    echo "[ERROR] At least one $label must be provided." >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase-label)
      PHASE_LABELS+=("$2")
      shift 2
      ;;
    --mp-material-id)
      MP_IDS+=("$2")
      shift 2
      ;;
    --local-cif)
      LOCAL_CIFS+=("$2")
      shift 2
      ;;
    --manual-reference)
      MANUAL_REFERENCES+=("$2")
      shift 2
      ;;
    --spectra-source)
      SPECTRA_SOURCE=$2
      shift 2
      ;;
    --run-name)
      RUN_NAME=$2
      shift 2
      ;;
    --num-spectra)
      NUM_SPECTRA=$2
      shift 2
      ;;
    --xrd-epochs)
      XRD_EPOCHS=$2
      shift 2
      ;;
    --pdf-epochs)
      PDF_EPOCHS=$2
      shift 2
      ;;
    --min-angle)
      MIN_ANGLE=$2
      shift 2
      ;;
    --max-angle)
      MAX_ANGLE=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_nonempty_list "--phase-label" "${#PHASE_LABELS[@]}"
if [[ "${#MP_IDS[@]}" -eq 0 && "${#LOCAL_CIFS[@]}" -eq 0 ]]; then
  echo "[ERROR] Provide at least one --mp-material-id or --local-cif." >&2
  usage >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_SLUG=""
  for label in "${PHASE_LABELS[@]}"; do
    if [[ -n "$RUN_SLUG" ]]; then
      RUN_SLUG+="-"
    fi
    RUN_SLUG+="$(slugify "$label")"
  done
  RUN_NAME="${RUN_SLUG}-$(date +%Y%m%d-%H%M%S)"
fi

CIF_RUN_DIR="$NOVEL_SPACE/soft_link/All_CIFs/$RUN_NAME"
SPECTRA_RUN_DIR="$NOVEL_SPACE/soft_link/Spectra/$RUN_NAME"
FIGURE_RUN_DIR="$NOVEL_SPACE/soft_link/figure/$RUN_NAME"
RESULTS_DIR="$CIF_RUN_DIR/results"

if [[ -e "$CIF_RUN_DIR" || -e "$SPECTRA_RUN_DIR" || -e "$FIGURE_RUN_DIR" ]]; then
  echo "[ERROR] Run name already exists: $RUN_NAME" >&2
  exit 1
fi

mkdir -p "$CIF_RUN_DIR" "$SPECTRA_RUN_DIR" "$FIGURE_RUN_DIR" "$RESULTS_DIR"
copy_spectra "$SPECTRA_SOURCE" "$SPECTRA_RUN_DIR"
link_workspace
mkdir -p "$FIGURE_RUN_DIR"

echo "[INFO] Starting container build/runtime..."
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose -f "$COMPOSE_FILE" up -d --build

if [[ "${#MP_IDS[@]}" -gt 0 ]]; then
  MP_ARGS=""
  for material_id in "${MP_IDS[@]}"; do
    MP_ARGS+=" --material-id $material_id"
  done
  echo "[INFO] Downloading MP CIFs..."
  compose_exec "cd $CONTAINER_ROOT && python3 $MP_TOOL download$MP_ARGS --output-dir $CONTAINER_NOVEL_SPACE/All_CIFs --manifest download_manifest.json"
fi

if [[ "${#LOCAL_CIFS[@]}" -gt 0 ]]; then
  echo "[INFO] Copying local CIFs..."
  for local_cif in "${LOCAL_CIFS[@]}"; do
    if [[ ! -f "$local_cif" ]]; then
      echo "[ERROR] Local CIF does not exist: $local_cif" >&2
      exit 1
    fi
    cp "$local_cif" "$CIF_RUN_DIR/"
  done
fi

echo "[INFO] Cleaning previous local Novel-Space state..."
rm -rf "$NOVEL_SPACE/References" "$NOVEL_SPACE/Models" "$NOVEL_SPACE/result.csv" \
  "$NOVEL_SPACE/XRD.npy" "$NOVEL_SPACE/PDF.npy" "$NOVEL_SPACE/angle_ranges.csv" \
  "$NOVEL_SPACE/Model.pth" "$NOVEL_SPACE/PDF_Model.pth"

if [[ "${#MANUAL_REFERENCES[@]}" -gt 0 ]]; then
  echo "[INFO] Creating manual References and skipping tabulate_cifs..."
  mkdir -p "$NOVEL_SPACE/References"
  for mapping in "${MANUAL_REFERENCES[@]}"; do
    ref_name=${mapping%%=*}
    src_name=${mapping#*=}
    if [[ -z "$ref_name" || -z "$src_name" || "$ref_name" == "$src_name" ]]; then
      echo "[ERROR] Invalid --manual-reference mapping: $mapping" >&2
      exit 1
    fi
    if [[ ! -f "$CIF_RUN_DIR/$src_name" ]]; then
      echo "[ERROR] Manual reference source CIF not found in run directory: $src_name" >&2
      exit 1
    fi
    cp "$CIF_RUN_DIR/$src_name" "$NOVEL_SPACE/References/${ref_name}.cif"
  done
fi

echo "[INFO] Training XRD model..."
XRD_ARGS="--num_spectra=$NUM_SPECTRA --num_epochs=$XRD_EPOCHS --min_angle=$MIN_ANGLE --max_angle=$MAX_ANGLE --save"
if [[ "${#MANUAL_REFERENCES[@]}" -gt 0 ]]; then
  XRD_ARGS+=" --skip_filter"
fi
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/construct_xrd_model.py $XRD_ARGS"
persist_dir_as_link "$NOVEL_SPACE/References" "$CIF_RUN_DIR/References"
[[ -f "$NOVEL_SPACE/XRD.npy" ]] && cp "$NOVEL_SPACE/XRD.npy" "$RESULTS_DIR/XRD.npy"
[[ -f "$CIF_RUN_DIR/download_manifest.json" ]] && cp "$CIF_RUN_DIR/download_manifest.json" "$RESULTS_DIR/download_manifest.json"

echo "[INFO] Training PDF model..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/construct_pdf_model.py --num_spectra=$NUM_SPECTRA --num_epochs=$PDF_EPOCHS --min_angle=$MIN_ANGLE --max_angle=$MAX_ANGLE"
persist_dir_as_link "$NOVEL_SPACE/Models" "$CIF_RUN_DIR/Models"
[[ -f "$NOVEL_SPACE/PDF.npy" ]] && cp "$NOVEL_SPACE/PDF.npy" "$RESULTS_DIR/PDF.npy"

echo "[INFO] Generating spectrum previews..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/plot_real_spectra.py && python3 src/extract_ranges.py"
[[ -f "$NOVEL_SPACE/angle_ranges.csv" ]] && cp "$NOVEL_SPACE/angle_ranges.csv" "$RESULTS_DIR/angle_ranges.csv"

echo "[INFO] Running inference..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/run_CNN.py --inc_pdf --show_indiv"
[[ -f "$NOVEL_SPACE/result.csv" ]] && cp "$NOVEL_SPACE/result.csv" "$RESULTS_DIR/result.csv"

{
  echo "run_name=$RUN_NAME"
  printf 'phase_labels=%s\n' "$(IFS=,; echo "${PHASE_LABELS[*]}")"
  printf 'mp_material_ids=%s\n' "$(IFS=,; echo "${MP_IDS[*]}")"
  printf 'local_cifs=%s\n' "$(IFS=,; echo "${LOCAL_CIFS[*]}")"
  printf 'manual_references=%s\n' "$(IFS=,; echo "${MANUAL_REFERENCES[*]}")"
  echo "spectra_source=$SPECTRA_SOURCE"
  echo "cif_dir=$CIF_RUN_DIR"
  echo "spectra_dir=$SPECTRA_RUN_DIR"
  echo "figure_dir=$FIGURE_RUN_DIR"
  echo "results_csv=$RESULTS_DIR/result.csv"
  echo "models_dir=$CIF_RUN_DIR/Models"
  echo "references_dir=$CIF_RUN_DIR/References"
} > "$RESULTS_DIR/run_manifest.txt"

echo "[DONE] Pipeline finished."
echo "run_name=$RUN_NAME"
echo "results_csv=$RESULTS_DIR/result.csv"
echo "models_dir=$CIF_RUN_DIR/Models"
echo "references_dir=$CIF_RUN_DIR/References"
echo "spectra_dir=$SPECTRA_RUN_DIR"
echo "figure_dir=$FIGURE_RUN_DIR"
