#!/usr/bin/env bash
# =====================================================================
# FilmForge - ONE command to do everything on a Lightning AI Studio.
# Installs ComfyUI + nodes, downloads the models, and launches ComfyUI.
#
#   HF_TOKEN=hf_your_token bash lightning_go.sh ltx23     # or wan22
# =====================================================================
set -e
MODEL="${1:-ltx23}"
HERE="$(cd "$(dirname "$0")" && pwd)"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
export COMFY_DIR

if [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN is not set. Run like:  HF_TOKEN=hf_xxx bash lightning_go.sh $MODEL"
  exit 1
fi

echo "########## 1/3  SETUP ##########"
bash "$HERE/lightning_setup.sh" "$MODEL"

# Reuse the exact Python that setup.sh located (conda env may not be on PATH).
[ -f "$HOME/.filmforge_env" ] && . "$HOME/.filmforge_env"
PY="$(command -v python || command -v python3)"

echo "########## 2/3  DOWNLOAD MODELS ##########"
"$PY" "$HERE/lightning_download.py" "$MODEL"

echo "########## 3/3  LAUNCH COMFYUI (port 8188) ##########"
echo "Open the Studio 'Ports' panel and expose port 8188 to use the UI."
cd "$COMFY_DIR"
exec "$PY" main.py --listen 0.0.0.0 --port 8188
