#!/usr/bin/env python3
"""
render_film.py - drive a LIVE ComfyUI (localhost:8188) to render every shot of
your film with its PER-CHARACTER LoRA, then stitch all clips into one movie.

This is the free-GPU / any-provider path (Lightning, E2E, AceCloud, Cyfuture,
AMD ...). It does NOT need Modal or Beam - it talks to whatever ComfyUI you
launched on the box.

WHAT IT DOES, PER SHOT
  1. picks the model (Wan 2.2 for short hero shots, LTX-2.3 for long/draft) via
     filmforge.choose_model - unless the shot sets "model".
  2. looks up the shot's "character" in the shots.json "characters" map to get a
     LoRA filename, builds a fresh graph with make_workflows.build(...) that
     stacks that LoRA (both experts for Wan), and injects prompt/neg/seed/size.
  3. POSTs the graph to ComfyUI /prompt, polls /history, downloads the mp4.
  Every shot of the same character uses the same LoRA => consistent identity
  across the final stitched film.
  4. stitches all clips with filmforge.stitch (hard cut or crossfade).

USAGE
  python3 render_film.py --shots shots.json --out film.mp4
  python3 render_film.py --shots shots.json --comfy http://127.0.0.1:8188 \
                         --transition crossfade --char-strength 0.85

SHOTS FILE (adds two things to the old format)
  {
    "fps": 24, "width": 768, "height": 448,
    "characters": { "hero": "hero_v1.safetensors" },   # name -> LoRA in models/loras
    "shots": [
      {"id":"01", "prompt":"wide neon city", "seconds":3},
      {"id":"02", "prompt":"close-up, eyes open", "seconds":3,
       "character":"hero", "char_strength":0.85}
    ]
  }
Shots with no "character" render with no character LoRA (scenery, etc.).
"""
import argparse, json, os, sys, time, uuid, pathlib, urllib.request, urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_workflows as mw            # graph builders (+char LoRA)
import filmforge as ff                 # choose_model + stitch (reuse)

CLIPS_DIR = pathlib.Path(os.path.expanduser("~")) / ".filmforge" / "clips"


def _api(base, path, data=None):
    url = base.rstrip("/") + path
    if data is not None:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _set_title(graph, title, key, value):
    """Set inputs[key]=value on every node whose _meta.title == title."""
    for node in graph.values():
        if node.get("_meta", {}).get("title") == title:
            node["inputs"][key] = value


def _set_seed(graph, seed):
    for node in graph.values():
        ct = node["class_type"]
        if ct == "KSampler":
            node["inputs"]["seed"] = seed
        elif ct == "KSamplerAdvanced":
            node["inputs"]["noise_seed"] = seed


def _set_prefix(graph, prefix):
    for node in graph.values():
        if node["class_type"] == "VHS_VideoCombine":
            node["inputs"]["filename_prefix"] = prefix


def render_shot(base, shot, spec, char_strength_default, upscale_default=False):
    fps = int(spec.get("fps", 24))
    width = int(spec.get("width", 768))
    height = int(spec.get("height", 448))
    characters = spec.get("characters", {}) or {}

    model = shot.get("model") or ff.choose_model(shot)
    seconds = float(shot.get("seconds", 5))
    cname = shot.get("character")
    char_lora = characters.get(cname) if cname else None
    if cname and not char_lora:
        print(f"  [warn] shot {shot.get('id')}: character '{cname}' not in "
              f"'characters' map; rendering with NO character LoRA")
    strength = float(shot.get("char_strength", char_strength_default))
    do_upscale = bool(shot.get("upscale", spec.get("upscale", upscale_default)))

    graph = mw.build(model, width, height, seconds,
                     char_lora=char_lora, char_strength=strength,
                     upscale=do_upscale)
    _set_title(graph, "POSITIVE_PROMPT", "text", shot["prompt"])
    _set_title(graph, "NEGATIVE_PROMPT", "text",
               shot.get("negative", mw.NEG))
    seed = int(shot.get("seed", 0)) or (int(time.time() * 1000) % (2 ** 31))
    _set_seed(graph, seed)
    prefix = f"ff_{shot.get('id', 'x')}"
    _set_prefix(graph, prefix)

    tag = f"{model}" + (f"+{cname}" if char_lora else "") + ("+4x" if do_upscale else "")
    print(f"  [shot {shot.get('id')}] {tag}  {seconds:g}s  seed={seed}")

    client_id = str(uuid.uuid4())
    resp = _api(base, "/prompt", {"prompt": graph, "client_id": client_id})
    pid = resp["prompt_id"]

    # poll history until this prompt finishes
    out_file = None
    while True:
        time.sleep(2)
        hist = _api(base, f"/history/{pid}")
        if pid not in hist:
            continue
        outputs = hist[pid].get("outputs", {})
        for node_out in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in node_out.get(key, []) or []:
                    if item.get("filename", "").endswith(".mp4"):
                        out_file = item
            if out_file:
                break
        status = hist[pid].get("status", {})
        if out_file or status.get("completed") or status.get("status_str") == "error":
            break
    if not out_file:
        raise RuntimeError(f"shot {shot.get('id')}: ComfyUI returned no mp4 "
                           f"(check the workflow/nodes on the server)")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    local = CLIPS_DIR / f"shot_{shot.get('id','x')}_{model}.mp4"
    q = urllib.parse.urlencode({"filename": out_file["filename"],
                                "subfolder": out_file.get("subfolder", ""),
                                "type": out_file.get("type", "output")})
    with urllib.request.urlopen(base.rstrip("/") + "/view?" + q, timeout=300) as r:
        local.write_bytes(r.read())
    print(f"    -> {local}")
    return str(local)


def main():
    ap = argparse.ArgumentParser(description="Render+stitch a film on a live ComfyUI")
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", default="film.mp4")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--transition", choices=["cut", "crossfade"], default="cut")
    ap.add_argument("--char-strength", type=float, default=mw.CHAR_STRENGTH)
    ap.add_argument("--upscale", action="store_true",
                    help="append 4x upscale to every shot (heavier; needs the model)")
    args = ap.parse_args()

    # Make the live graph reference the EXACT model filenames on disk (from the
    # downloader's manifest), not the hardcoded defaults. No-op if absent.
    applied = mw.apply_manifest()
    if applied:
        print(f"[manifest] model filenames from disk: {applied}")

    spec = json.loads(pathlib.Path(args.shots).read_text())
    shots = spec["shots"]
    chars = spec.get("characters", {}) or {}
    print(f"render_film: {len(shots)} shots via {args.comfy} | "
          f"characters={list(chars) or 'none'} | transition={args.transition}\n")

    # sanity: warn about missing LoRA files if we can see the models dir
    comfy_dir = os.environ.get("COMFY_DIR", os.path.expanduser("~/ComfyUI"))
    loras_dir = pathlib.Path(comfy_dir) / "models" / "loras"
    for name, f in chars.items():
        if loras_dir.exists() and not (loras_dir / f).exists():
            print(f"  [warn] character '{name}' LoRA not found: {loras_dir / f}")

    clips = []
    for shot in shots:
        clips.append(render_shot(args.comfy, shot, spec,
                                 args.char_strength, args.upscale))

    print(f"\nStitching {len(clips)} clips -> {args.out} ({args.transition})")
    ff.stitch(clips, args.out, transition=args.transition)
    print(f"Done. Final film: {args.out} ({ff._duration(args.out):.2f}s)")


if __name__ == "__main__":
    main()
