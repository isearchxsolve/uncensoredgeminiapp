#!/usr/bin/env python3
"""
filmforge.py — end-to-end AI film pipeline (single file)
=======================================================
One script that does everything discussed:

  1. Reads a SHOT LIST (JSON) describing your film shot by shot.
  2. Picks the right MODEL per shot:
        - Wan 2.2   -> short hero / max-quality shots
        - LTX-2.3   -> long shots (>8s), dialogue/audio, or draft passes
  3. Generates each clip through a CHAIN OF GPU PROVIDERS with automatic
     FAILOVER (Modal -> Beam -> ...), tracking monthly free-credit usage so it
     switches to the next provider BEFORE one free tier is burned.
  4. STITCHES all clips into one film with ffmpeg (hard cut or crossfade,
     audio-aware).

Runnable RIGHT NOW in --dry-run mode (no cloud, no GPU, no network): it renders
coloured placeholder clips WITH silent audio via ffmpeg and stitches them, so
you can validate the entire pipeline offline. Turn off --dry-run once your
Modal / Beam backends are deployed.

--------------------------------------------------------------------------------
USAGE
  python3 filmforge.py --shots shots.json --out film.mp4 --transition crossfade
  python3 filmforge.py --shots shots.json --dry-run          # offline test
  python3 filmforge.py --status                              # usage ledger

SHOTS FILE (shots.json)
  {
    "fps": 24, "width": 1216, "height": 704,
    "shots": [
      {"id": "01", "prompt": "wide neon city at dawn", "seconds": 6,
       "quality": "balanced"},
      {"id": "02", "prompt": "close-up, hero's eyes open", "seconds": 3,
       "quality": "max", "images": ["hero.png"]},
      {"id": "03", "prompt": "hero walks into frame", "seconds": 8,
       "quality": "draft", "video": "prev.mp4"}
    ]
  }

Provider backends are the deployed ComfyUI apps from the setup guide:
  - Modal:  a deployed app per model, class "ComfyUI", method generate(payload)
  - Beam :  an HTTPS endpoint that accepts the same payload and returns mp4 bytes
--------------------------------------------------------------------------------
"""

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

# =============================================================================
# 0. CONFIG
# =============================================================================
HOME_DIR = pathlib.Path(os.path.expanduser("~")) / ".filmforge"
LEDGER = HOME_DIR / "usage.json"
CLIPS_DIR = HOME_DIR / "clips"

FPS_DEFAULT = 24

# GPU-seconds burned per output-second (empirical rough estimates).
GPU_SECONDS_PER_OUTPUT_SECOND = {"wan22": 55.0, "ltx23": 18.0}

# Provider chain, tried in order. $ budgets are the free monthly credit tiers.
# cost_per_gpu_s = approximate $ per GPU-second on that provider.
PROVIDERS = [
    {"name": "modal", "budget_usd": 30.0, "cost_per_gpu_s": 0.00110,
     "kind": "modal"},
    {"name": "beam",  "budget_usd": 30.0, "cost_per_gpu_s": 0.00105,
     "kind": "beam",  "endpoint_env": "BEAM_ENDPOINT", "token_env": "BEAM_TOKEN"},
]


class QuotaExhausted(Exception):
    """Raised when a provider's projected spend would exceed its free budget."""


# =============================================================================
# 1. USAGE LEDGER  (monthly, auto-resets)
# =============================================================================
def _month_key():
    return dt.datetime.utcnow().strftime("%Y-%m")


def load_ledger():
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        return {}
    try:
        data = json.loads(LEDGER.read_text())
    except Exception:
        data = {}
    return data.get(_month_key(), {})


def save_ledger(month_data):
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    full = {}
    if LEDGER.exists():
        try:
            full = json.loads(LEDGER.read_text())
        except Exception:
            full = {}
    full[_month_key()] = month_data
    LEDGER.write_text(json.dumps(full, indent=2))


def record_spend(provider_name, usd):
    month = load_ledger()
    month[provider_name] = round(month.get(provider_name, 0.0) + usd, 6)
    save_ledger(month)


def spent(provider_name):
    return load_ledger().get(provider_name, 0.0)


def remaining(provider):
    return provider["budget_usd"] - spent(provider["name"])


def estimate_cost(provider, model, seconds):
    gpu_s = GPU_SECONDS_PER_OUTPUT_SECOND[model] * seconds
    return gpu_s * provider["cost_per_gpu_s"]


# =============================================================================
# 2. MODEL SELECTION  (Wan vs LTX)
# =============================================================================
def choose_model(shot):
    """LTX-2.3 for long / draft / dialogue-audio shots; Wan 2.2 for short hero."""
    seconds = float(shot.get("seconds", 5))
    quality = str(shot.get("quality", "balanced")).lower()
    if shot.get("model"):                       # explicit override wins
        return shot["model"]
    if quality == "draft" or seconds > 8:
        return "ltx23"
    if quality == "max":
        return "wan22"
    return "wan22" if seconds <= 5 else "ltx23"  # balanced: short->Wan, longer->LTX


# =============================================================================
# 3. PROVIDERS  (each returns a local mp4 path)
# =============================================================================
def _payload_for(shot, model, fps, width, height):
    def _b64(path):
        return base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    payload = {
        "model": model,
        "prompt": shot["prompt"],
        "negative": shot.get("negative", "blurry, distorted, low quality"),
        "seconds": float(shot.get("seconds", 5)),
        "seed": int(shot.get("seed", 0)) or int(time.time() * 1000) % (2 ** 31),
        "fps": fps, "width": width, "height": height,
        "images": [], "video": None,
    }
    for img in shot.get("images", []) or []:
        payload["images"].append((os.path.basename(img), _b64(img)))
    if shot.get("video"):
        payload["video"] = (os.path.basename(shot["video"]), _b64(shot["video"]))
    return payload


def gen_modal(provider, shot, model, fps, width, height, out_path):
    """Call a deployed Modal app: app name f\"{model}-comfyui\", class ComfyUI."""
    import modal  # lazy: only needed in real mode
    payload = _payload_for(shot, model, fps, width, height)
    Comfy = modal.Cls.lookup(f"{model}-comfyui", "ComfyUI")
    data = Comfy().generate.remote(payload)
    pathlib.Path(out_path).write_bytes(data)
    return out_path


def gen_beam(provider, shot, model, fps, width, height, out_path):
    """POST payload to a Beam HTTPS endpoint; expect raw mp4 bytes back."""
    endpoint = os.environ.get(provider["endpoint_env"])
    token = os.environ.get(provider["token_env"], "")
    if not endpoint:
        raise RuntimeError(f"{provider['endpoint_env']} not set")
    payload = _payload_for(shot, model, fps, width, height)
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (402, 429):                # payment required / rate limited
            raise QuotaExhausted(f"beam HTTP {e.code}")
        raise
    pathlib.Path(out_path).write_bytes(data)
    return out_path


def gen_dryrun(provider, shot, model, fps, width, height, out_path):
    """Offline placeholder: solid-colour clip + label + silent audio via ffmpeg."""
    seconds = float(shot.get("seconds", 5))
    colors = {"wan22": "purple", "ltx23": "teal"}
    color = colors.get(model, "gray")
    # Solid-colour placeholder + silent audio (drawtext omitted: not in every
    # ffmpeg build). Colour encodes which model would have run the shot.
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:d={seconds}:r={fps}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


GENERATORS = {"modal": gen_modal, "beam": gen_beam}


# =============================================================================
# 4. ORCHESTRATION  (failover across providers)
# =============================================================================
def generate_shot(shot, fps, width, height, dry_run=False):
    model = choose_model(shot)
    seconds = float(shot.get("seconds", 5))
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(CLIPS_DIR / f"shot_{shot.get('id','x')}_{model}.mp4")

    if dry_run:
        print(f"  [dry-run] shot {shot.get('id')} -> {model} ({seconds:g}s)")
        return gen_dryrun(None, shot, model, fps, width, height, out_path), model

    last_err = None
    for provider in PROVIDERS:
        cost = estimate_cost(provider, model, seconds)
        if cost > remaining(provider):
            print(f"  [skip] {provider['name']}: est ${cost:.3f} > "
                  f"remaining ${remaining(provider):.3f}")
            continue
        try:
            print(f"  [try ] {provider['name']} for shot {shot.get('id')} "
                  f"({model}, est ${cost:.3f})")
            path = GENERATORS[provider["kind"]](
                provider, shot, model, fps, width, height, out_path)
            record_spend(provider["name"], cost)
            print(f"  [ok  ] {provider['name']} -> {path}")
            return path, model
        except QuotaExhausted as e:
            print(f"  [quota] {provider['name']} exhausted: {e}; failing over")
            record_spend(provider["name"], remaining(provider))  # burn the rest
            last_err = e
        except Exception as e:                  # network / runtime -> next provider
            print(f"  [err ] {provider['name']} failed: {e}; failing over")
            last_err = e
    raise RuntimeError(f"All providers failed for shot {shot.get('id')}: {last_err}")


# =============================================================================
# 5. STITCHING  (ffmpeg; hard cut or crossfade, audio-aware)
# =============================================================================
def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


def stitch_cut(clips, out_path):
    """Concatenate with hard cuts (re-encode for a uniform stream)."""
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for c in clips:
        lst.write(f"file '{os.path.abspath(c)}'\n")
    lst.close()
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst.name,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(lst.name)
    return out_path


def _has_audio(path):
    """True only if the file actually contains an audio stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    return bool(out)


def stitch_crossfade(clips, out_path, transition="fade", dur=0.5):
    """Crossfade video (and audio when present) between consecutive clips.
    Rendered clips are usually SILENT - the LTX/Wan graphs do not wire audio
    into VHS_VideoCombine - so the audio crossfade is applied ONLY when EVERY
    clip actually has an audio stream. Otherwise we build a clean video-only
    mix (mapping a non-existent [i:a] is what previously crashed the stitch)."""
    if len(clips) == 1:
        return stitch_cut(clips, out_path)
    durs = [_duration(c) for c in clips]
    have_audio = all(_has_audio(c) for c in clips)
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    vfilters, afilters = [], []
    # normalise every input first (same size/fps/sar) so xfade won't error
    for i in range(len(clips)):
        vfilters.append(f"[{i}:v]settb=AVTB,fps=30,format=yuv420p,setsar=1[v{i}]")
        if have_audio:
            afilters.append(f"[{i}:a]aresample=48000[a{i}]")
    vlast, alast = "v0", "a0"
    offset = 0.0
    chain = []
    for i in range(1, len(clips)):
        offset += durs[i - 1] - dur
        vout = f"vx{i}"
        chain.append(
            f"[{vlast}][v{i}]xfade=transition={transition}:duration={dur}:"
            f"offset={offset:.3f}[{vout}]")
        vlast = vout
        if have_audio:
            aout = f"ax{i}"
            chain.append(f"[{alast}][a{i}]acrossfade=d={dur}[{aout}]")
            alast = aout
    filter_complex = ";".join(vfilters + afilters + chain)
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", f"[{vlast}]"]
    if have_audio:
        cmd += ["-map", f"[{alast}]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def stitch(clips, out_path, transition="cut"):
    if not clips:
        raise RuntimeError("no clips to stitch")
    if transition == "crossfade":
        return stitch_crossfade(clips, out_path)
    return stitch_cut(clips, out_path)


# =============================================================================
# 6. CLI
# =============================================================================
def cmd_status():
    month = load_ledger()
    print(f"FilmForge usage ledger for {_month_key()} (UTC):")
    if not month:
        print("  (nothing spent yet this month)")
    for p in PROVIDERS:
        print(f"  {p['name']:6s}  spent ${spent(p['name']):.3f} / "
              f"${p['budget_usd']:.2f}  (remaining ${remaining(p):.3f})")


def run(shots_file, out_path, transition, dry_run):
    spec = json.loads(pathlib.Path(shots_file).read_text())
    fps = int(spec.get("fps", FPS_DEFAULT))
    width = int(spec.get("width", 1216))
    height = int(spec.get("height", 704))
    shots = spec["shots"]
    print(f"FilmForge: {len(shots)} shots | {width}x{height}@{fps} | "
          f"transition={transition} | dry_run={dry_run}\n")

    clips = []
    for shot in shots:
        path, model = generate_shot(shot, fps, width, height, dry_run=dry_run)
        clips.append(path)
    print(f"\nStitching {len(clips)} clips -> {out_path} ({transition})")
    stitch(clips, out_path, transition=transition)
    total = _duration(out_path)
    print(f"Done. Final film: {out_path} ({total:.2f}s)")
    if not dry_run:
        cmd_status()


def main():
    ap = argparse.ArgumentParser(description="FilmForge end-to-end AI film pipeline")
    ap.add_argument("--shots", help="path to shots.json")
    ap.add_argument("--out", default="film.mp4", help="output film path")
    ap.add_argument("--transition", choices=["cut", "crossfade"], default="cut")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline placeholder clips (no cloud/GPU)")
    ap.add_argument("--status", action="store_true", help="show usage ledger")
    args = ap.parse_args()

    if args.status:
        cmd_status()
        return
    if not args.shots:
        ap.error("--shots is required (or use --status)")
    run(args.shots, args.out, args.transition, args.dry_run)


if __name__ == "__main__":
    main()
