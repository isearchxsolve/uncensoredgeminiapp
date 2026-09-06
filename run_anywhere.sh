#!/usr/bin/env bash
# =====================================================================
# FilmForge - PORTABLE one-command deploy for ANY Linux + NVIDIA + SSH box
# (Lightning, E2E, AceCloud, Cyfuture, RunPod, Vast, AMD w/ tweaks ...).
#
# Same pipeline everywhere. Switching providers = SSH into the new box,
# copy this folder, run ONE command below. No per-provider rebuild.
#
#   HF_TOKEN=hf_xxx bash run_anywhere.sh all    ltx23   # setup+dl+serve+render
#   HF_TOKEN=hf_xxx bash run_anywhere.sh setup  ltx23   # install + download only
#   bash run_anywhere.sh serve  ltx23                   # just launch ComfyUI
#   bash run_anywhere.sh render ltx23                    # render film (ComfyUI up)
#
# Optional env:
#   COMFY_DIR   where ComfyUI lives           (default ~/ComfyUI)
#   SHOTS       shot list for render          (default shots.json)
#   OUT         output film                    (default film.mp4)
#   TRANSITION  cut | crossfade               (default cut)
#   PORT        ComfyUI port                    (default 8188)
# =====================================================================
set -e
CMD="${1:-all}"
MODEL="${2:-ltx23}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
SHOTS="${SHOTS:-$HERE/shots.json}"
OUT="${OUT:-film.mp4}"
TRANSITION="${TRANSITION:-cut}"
PORT="${PORT:-8188}"
BASE="http://127.0.0.1:$PORT"

# --- find a Python interpreter (conda env may be off PATH over SSH) ----
find_py() {
  PY="$(command -v python || command -v python3 || true)"
  if [ -z "$PY" ]; then
    for f in /home/zeus/miniconda3/envs/*/bin/python "$HOME"/miniconda3/envs/*/bin/python \
             /opt/conda/envs/*/bin/python /opt/conda/bin/python \
             /usr/local/bin/python3 /usr/bin/python3; do
      [ -x "$f" ] && { PY="$f"; break; }
    done
  fi
  [ -z "$PY" ] && { echo "[X] No Python found"; exit 1; }
  export PATH="$(dirname "$PY"):$PATH"
}

wait_for_comfy() {
  echo "waiting for ComfyUI on $BASE ..."
  for i in $(seq 1 90); do
    if curl -sf "$BASE/system_stats" >/dev/null 2>&1; then echo "ComfyUI is up."; return 0; fi
    sleep 2
  done
  echo "[X] ComfyUI did not come up on $BASE"; return 1
}

do_setup() {
  bash "$HERE/lightning_setup.sh" "$MODEL"
  [ -f "$HOME/.filmforge_env" ] && . "$HOME/.filmforge_env"
  find_py
  [ -z "$HF_TOKEN" ] && { echo "HF_TOKEN not set; needed to download models"; exit 1; }
  "$PY" "$HERE/lightning_download.py" "$MODEL"
  # (re)generate baseline workflows (no character) for the UI
  ( cd "$HERE" && "$PY" make_workflows.py --force >/dev/null 2>&1 || true )
}

do_serve() {   # blocking
  find_py
  cd "$COMFY_DIR"
  exec "$PY" main.py --listen 0.0.0.0 --port "$PORT"
}

do_render() {
  find_py
  "$PY" "$HERE/render_film.py" --shots "$SHOTS" --out "$OUT" \
        --comfy "$BASE" --transition "$TRANSITION"
}

case "$CMD" in
  setup)  do_setup ;;
  serve)  do_serve ;;
  render) do_render ;;
  all)
    do_setup
    find_py
    echo "launching ComfyUI in background (log: $HERE/comfy.log)"
    ( cd "$COMFY_DIR" && "$PY" main.py --listen 0.0.0.0 --port "$PORT" \
        > "$HERE/comfy.log" 2>&1 & echo $! > "$HERE/comfy.pid" )
    wait_for_comfy
    do_render
    echo "ComfyUI still running (pid $(cat "$HERE/comfy.pid")). Stop it with: kill \$(cat $HERE/comfy.pid)"
    echo "IMPORTANT: on a paid/credit GPU, STOP the machine now to stop billing."
    ;;
  *) echo "usage: bash run_anywhere.sh {all|setup|serve|render} [ltx23|wan22]"; exit 1 ;;
esac
