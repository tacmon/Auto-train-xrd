#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yaml"
SERVICE_NAME="xrd-service"
CONTAINER_ROOT="/workspace/project"
NOVEL_SPACE="$REPO_ROOT/libs/XRD-1.1/Novel-Space"
CONTAINER_NOVEL_SPACE="$CONTAINER_ROOT/libs/XRD-1.1/Novel-Space"
SKILL_TOOL="$CONTAINER_ROOT/.codex/skills/xrd-formula-train-infer/scripts/mp_formula_tool.py"

FORMULA_A=""
FORMULA_B=""
MATERIAL_ID_A=""
MATERIAL_ID_B=""
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
  run_pipeline.sh --formula-a A --formula-b B --material-id-a mp-xxx --material-id-b mp-yyy [options]

Options:
  --spectra-source PATH   Real spectra source. Default: ./data
  --run-name NAME         Explicit run directory name
  --num-spectra N         Simulated spectra per phase. Default: 50
  --xrd-epochs N          XRD training epochs. Default: 50
  --pdf-epochs N          PDF training epochs. Default: 50
  --min-angle VALUE       Lower two-theta bound. Default: 20.0
  --max-angle VALUE       Upper two-theta bound. Default: 60.0
EOF
}

require_arg() {
  local name=$1
  local value=$2
  if [[ -z "$value" ]]; then
    echo "[ERROR] Missing required argument: $name" >&2
    usage >&2
    exit 1
  fi
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

compose_exec() {
  docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" bash -lc "$1"
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
  local cif_dir=$1
  local spectra_dir=$2
  local figure_dir=$3

  mkdir -p "$NOVEL_SPACE/figure"
  rm -rf "$NOVEL_SPACE/Spectra" "$NOVEL_SPACE/All_CIFs" "$NOVEL_SPACE/Models" "$NOVEL_SPACE/References" "$NOVEL_SPACE/figure/real_data"
  ln -snf "$spectra_dir" "$NOVEL_SPACE/Spectra"
  ln -snf "$cif_dir" "$NOVEL_SPACE/All_CIFs"
  ln -snf "../soft_link/figure/$RUN_NAME" "$NOVEL_SPACE/figure/real_data"
  mkdir -p "$figure_dir"
}

persist_dir_as_link() {
  local local_dir=$1
  local target_dir=$2
  if [[ -d "$local_dir" && ! -L "$local_dir" ]]; then
    mv "$local_dir" "$target_dir"
    ln -snf "$target_dir" "$local_dir"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formula-a)
      FORMULA_A=$2
      shift 2
      ;;
    --formula-b)
      FORMULA_B=$2
      shift 2
      ;;
    --material-id-a)
      MATERIAL_ID_A=$2
      shift 2
      ;;
    --material-id-b)
      MATERIAL_ID_B=$2
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

require_arg "--formula-a" "$FORMULA_A"
require_arg "--formula-b" "$FORMULA_B"
require_arg "--material-id-a" "$MATERIAL_ID_A"
require_arg "--material-id-b" "$MATERIAL_ID_B"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(slugify "$FORMULA_A")-vs-$(slugify "$FORMULA_B")-$(date +%Y%m%d-%H%M%S)"
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
link_workspace "$CIF_RUN_DIR" "$SPECTRA_RUN_DIR" "$FIGURE_RUN_DIR"

echo "[INFO] Starting container build/runtime..."
docker compose -f "$COMPOSE_FILE" up -d --build

echo "[INFO] Downloading CIFs for $MATERIAL_ID_A and $MATERIAL_ID_B ..."
compose_exec "cd $CONTAINER_ROOT && python3 $SKILL_TOOL download --material-id $MATERIAL_ID_A --material-id $MATERIAL_ID_B --output-dir $CONTAINER_NOVEL_SPACE/All_CIFs --manifest download_manifest.json"

echo "[INFO] Cleaning previous local Novel-Space state..."
rm -rf "$NOVEL_SPACE/References" "$NOVEL_SPACE/Models" "$NOVEL_SPACE/result.csv" "$NOVEL_SPACE/XRD.npy" "$NOVEL_SPACE/PDF.npy" "$NOVEL_SPACE/angle_ranges.csv"

echo "[INFO] Training XRD model..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/construct_xrd_model.py --num_spectra=$NUM_SPECTRA --num_epochs=$XRD_EPOCHS --min_angle=$MIN_ANGLE --max_angle=$MAX_ANGLE --save"
persist_dir_as_link "$NOVEL_SPACE/References" "$CIF_RUN_DIR/References"
[[ -f "$NOVEL_SPACE/XRD.npy" ]] && cp "$NOVEL_SPACE/XRD.npy" "$RESULTS_DIR/XRD.npy"
[[ -f "$CIF_RUN_DIR/download_manifest.json" ]] && cp "$CIF_RUN_DIR/download_manifest.json" "$RESULTS_DIR/download_manifest.json"

echo "[INFO] Training PDF model..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/construct_pdf_model.py --num_spectra=$NUM_SPECTRA --num_epochs=$PDF_EPOCHS --min_angle=$MIN_ANGLE --max_angle=$MAX_ANGLE"
persist_dir_as_link "$NOVEL_SPACE/Models" "$CIF_RUN_DIR/Models"

echo "[INFO] Generating spectrum previews..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/plot_real_spectra.py && python3 src/extract_ranges.py"
[[ -f "$NOVEL_SPACE/angle_ranges.csv" ]] && cp "$NOVEL_SPACE/angle_ranges.csv" "$RESULTS_DIR/angle_ranges.csv"

echo "[INFO] Running inference..."
compose_exec "cd $CONTAINER_NOVEL_SPACE && python3 src/run_CNN.py --inc_pdf --show_indiv"
[[ -f "$NOVEL_SPACE/result.csv" ]] && cp "$NOVEL_SPACE/result.csv" "$RESULTS_DIR/result.csv"

cat > "$RESULTS_DIR/run_manifest.txt" <<EOF
run_name=$RUN_NAME
formula_a=$FORMULA_A
formula_b=$FORMULA_B
material_id_a=$MATERIAL_ID_A
material_id_b=$MATERIAL_ID_B
spectra_source=$SPECTRA_SOURCE
cif_dir=$CIF_RUN_DIR
spectra_dir=$SPECTRA_RUN_DIR
figure_dir=$FIGURE_RUN_DIR
results_csv=$RESULTS_DIR/result.csv
models_dir=$CIF_RUN_DIR/Models
references_dir=$CIF_RUN_DIR/References
EOF

echo "[DONE] Pipeline finished."
echo "run_name=$RUN_NAME"
echo "results_csv=$RESULTS_DIR/result.csv"
echo "models_dir=$CIF_RUN_DIR/Models"
echo "references_dir=$CIF_RUN_DIR/References"
echo "spectra_dir=$SPECTRA_RUN_DIR"
echo "figure_dir=$FIGURE_RUN_DIR"
