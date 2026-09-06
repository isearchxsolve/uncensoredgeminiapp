#!/usr/bin/env bash
# ===========================================================================
# FilmForge GPU BURST - run this on the Lightning Studio (after SSH).
# It does the whole cheap first-win in one shot:
#   1. start ComfyUI (python3 fallback, logs to ~/comfy.log)
#   2. wait until it actually answers (never renders blind)
#   3. run the ~1-second SMOKE render (your first real proof clip)
#   4. remind you to STOP THE STUDIO so credits stop burning
#
# Nothing here burns meaningful GPU except the ~1s smoke render. Everything
# heavy stays off until you decide to scale up.
# ===========================================================================
set -o pipefail

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
FF_DIR="${FF_DIR:-$HOME}"          # where make_workflows.py + preflight.py live
PORT="${PORT:-8188}"

# --- quality knobs (safe defaults; override BEFORE running if you want) -----
export FF_QUANT="${FF_QUANT:-q5_k_m}"     # q4_k_m | q5_k_m | q6_k
export FF_TILED_VAE="${FF_TILED_VAE:-1}"  # avoids OOM at higher precision

# --- find a python ----------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
echo "[burst] python = $PY   quant = $FF_QUANT   tiled_vae = $FF_TILED_VAE"

# --- 1) start ComfyUI (skip if it is already answering) ---------------------
if curl -sf "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  echo "[burst] ComfyUI already up on port $PORT"
else
  echo "[burst] starting ComfyUI from $COMFY_DIR ..."
  ( cd "$COMFY_DIR" && source "$HOME/.filmforge_env" 2>/dev/null; \
    nohup "$PY" main.py --listen 0.0.0.0 --port "$PORT" > "$HOME/comfy.log" 2>&1 & )
fi

# --- 2) wait up to ~120s for readiness --------------------------------------
printf "[burst] waiting for ComfyUI to answer"
UP=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then UP=1; echo " - UP"; break; fi
  printf "."; sleep 2
done
if [ "$UP" != "1" ]; then
  echo
  echo "[burst] ComfyUI did not answer in time. Last lines of ~/comfy.log:"
  echo "-------------------------------------------------------------------"
  tail -n 40 "$HOME/comfy.log" 2>/dev/null || echo "(no log found)"
  echo "-------------------------------------------------------------------"
  echo "[burst] Fix the error above, then re-run:  bash ~/go_burst.sh"
  exit 1
fi

# --- 3) SMOKE render (tiny 1s clip = your first real proof) -----------------
if [ -f "$FF_DIR/preflight.py" ] && [ -f "$FF_DIR/make_workflows.py" ]; then
  echo "[burst] building workflows + running smoke render ..."
  ( cd "$FF_DIR" && "$PY" make_workflows.py --force >/dev/null 2>&1; \
    "$PY" preflight.py --comfy "http://127.0.0.1:$PORT" --smoke )
  SMOKE_RC=$?
else
  echo "[burst] preflight.py / make_workflows.py not found in $FF_DIR."
  echo "        From your Windows PC (Downloads\\filmforge) run this ONCE, then re-run me:"
  echo "        scp make_workflows.py preflight.py render_film.py shots.json go_burst.sh \\"
  echo "            $(whoami)@ssh.lightning.ai:~/"
  SMOKE_RC=2
fi

# --- 4) STOP reminder -------------------------------------------------------
echo
echo "==================================================================="
if [ "${SMOKE_RC:-1}" = "0" ]; then
  echo " SMOKE PASSED - your first clip is in:  $COMFY_DIR/output/"
  echo " (look for preflight_smoke*.mp4)"
fi
echo " NEXT: open the tunnel URL  http://localhost:$PORT  to use the UI,"
echo " OR render a full film:   cd $FF_DIR && $PY render_film.py --shots shots.json"
echo
echo " >>> WHEN YOU ARE DONE, STOP THE STUDIO IN THE LIGHTNING UI <<<"
echo "     A stopped Studio = ZERO GPU credits. Your files are kept."
echo "==================================================================="
