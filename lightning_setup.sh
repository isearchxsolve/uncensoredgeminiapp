#!/usr/bin/env bash
# =====================================================================
# FilmForge - one-time ComfyUI setup on a Lightning AI Studio
# Run this ONCE inside a Studio terminal. It persists on the Studio disk.
#
#   bash lightning_setup.sh ltx23     # or: bash lightning_setup.sh wan22
# =====================================================================
set -e
MODEL="${1:-ltx23}"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"

# ---------------------------------------------------------------------
# Locate a Python interpreter. Over a NON-interactive SSH session the
# Studio's conda env is often NOT on PATH, and a bare `pip` may not exist
# at all. So find python explicitly, add it to PATH, and remember it in
# ~/.filmforge_env so the later download + launch steps reuse the exact
# same interpreter. Always call pip as `"$PY" -m pip`, never bare `pip`.
# ---------------------------------------------------------------------
PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then
  for f in /home/zeus/miniconda3/envs/*/bin/python /home/zeus/miniconda3/bin/python \
           "$HOME"/miniconda3/envs/*/bin/python "$HOME"/miniconda3/bin/python \
           /opt/conda/envs/*/bin/python /opt/conda/bin/python \
           /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$f" ]; then PY="$f"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "[X] No Python interpreter found on the Studio. Aborting."
  exit 1
fi
export PATH="$(dirname "$PY"):$PATH"
printf 'export PATH="%s:$PATH"\n' "$(dirname "$PY")" > "$HOME/.filmforge_env"
echo "Using Python: $PY"

echo "== [1/4] Installing ComfyUI into $COMFY_DIR =="
if [ ! -d "$COMFY_DIR" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
fi
cd "$COMFY_DIR"
"$PY" -m pip install -r requirements.txt

echo "== [2/4] Installing the HuggingFace downloader =="
"$PY" -m pip install "huggingface_hub[hf_transfer]"

echo "== [3/4] Installing custom nodes for $MODEL =="
mkdir -p "$COMFY_DIR/custom_nodes"
cd "$COMFY_DIR/custom_nodes"
clone() { [ -d "$(basename "$1")" ] || git clone --depth 1 "$1"; }
clone https://github.com/ltdrdata/ComfyUI-Manager
clone https://github.com/city96/ComfyUI-GGUF
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
clone https://github.com/kijai/ComfyUI-KJNodes
if [ "$MODEL" = "ltx23" ]; then
  clone https://github.com/pollockjj/ComfyUI-MultiGPU
  clone https://github.com/Lightricks/ComfyUI-LTXVideo
fi

echo "== [4/4] Installing custom-node requirements =="
for d in */ ; do
  [ -f "$d/requirements.txt" ] && ("$PY" -m pip install -r "$d/requirements.txt" || true)
done

echo ""
echo "== DONE. ComfyUI + nodes installed for $MODEL (python: $PY) =="
