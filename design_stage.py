#!/usr/bin/env python3
"""
design_stage.py - the DESIGN front-end for FilmForge's per-character workflow.

With only 2-3 leads per film, every character is a lead worth a full LoRA. The
flow per character is:

    DESIGN  ->  TRAIN  ->  RENDER
    (lock a       (LoRA on     (shots.json
     canonical     that         loads the
     still)        design)      LoRA)

This tool manages characters.json (the design bible), optionally generates each
lead's canonical reference still on a live ComfyUI, tracks readiness, prints the
LoRA training command, and syncs the name->LoRA map into shots.json so
render_film.py picks it up. Nothing here needs a GPU except 'gen-ref'.

SUBCOMMANDS
  init                 write a template characters.json
  add <id>             add/replace a character entry
  status               readiness table + the NEXT ACTION per character
  gen-ref <id>         render the canonical still on a live ComfyUI (SDXL T2I)
  sync-shots           push {id -> lora} from the bible into shots.json
  train-cmd <id>       print a ready-to-run LoRA training command

EXAMPLES
  python3 design_stage.py init
  python3 design_stage.py add meera --name Meera \
      --prompt "cinematic portrait of a woman late 20s, beauty mark above left brow, copper hair streak, teal kurta"
  python3 design_stage.py gen-ref meera --comfy http://127.0.0.1:8188 --ckpt realvisxlV50.safetensors
  python3 design_stage.py status
  python3 design_stage.py train-cmd meera
  python3 design_stage.py sync-shots --shots shots.json
"""
import argparse, json, os, pathlib, sys, time, uuid, urllib.request, urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_FILE = "characters.json"


# --------------------------------------------------------------------------- io
def load_bible(path):
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"[design] {path} not found - run: design_stage.py init")
    return json.loads(p.read_text())


def save_bible(path, data):
    pathlib.Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _comfy_dir():
    return os.environ.get("COMFY_DIR", os.path.expanduser("~/ComfyUI"))


def _loras_dir():
    return pathlib.Path(_comfy_dir()) / "models" / "loras"


TEMPLATE = {
    "_comment": "DESIGN BIBLE. Flow per lead: DESIGN -> TRAIN -> RENDER. "
                "Run design_stage.py sync-shots to push name->LoRA into shots.json.",
    "image_model": "sd_xl_base_1.0.safetensors",
    "ref_width": 832,
    "ref_height": 1216,
    "characters": {},
}


# ----------------------------------------------------------------- subcommands
def cmd_init(args):
    p = pathlib.Path(args.file)
    if p.exists() and not args.force:
        sys.exit(f"[design] {args.file} already exists (use --force to overwrite)")
    save_bible(args.file, TEMPLATE)
    print(f"[design] wrote template {args.file}")


def cmd_add(args):
    bible = load_bible(args.file)
    chars = bible.setdefault("characters", {})
    entry = chars.get(args.id, {})
    entry.update({
        "display_name": args.name or entry.get("display_name") or args.id.title(),
        "tier": args.tier or entry.get("tier", "lead"),
        "design_prompt": args.prompt if args.prompt is not None else entry.get("design_prompt", ""),
        "negative": args.negative if args.negative is not None else entry.get(
            "negative", "blurry, distorted, low quality, watermark, text, deformed"),
        "reference_image": entry.get("reference_image", f"characters/{args.id}_ref.png"),
        "lora": args.lora or entry.get("lora", f"{args.id}_v1.safetensors"),
        "lora_strength": args.strength if args.strength is not None else entry.get("lora_strength", 0.85),
        "seed": args.seed if args.seed is not None else entry.get("seed", 0),
    })
    chars[args.id] = entry
    save_bible(args.file, bible)
    print(f"[design] saved character '{args.id}' -> {args.file}")


def _has_ref(entry):
    ref = entry.get("reference_image")
    return bool(ref) and pathlib.Path(ref).exists()


def _has_lora(entry):
    ld = _loras_dir()
    return ld.exists() and (ld / entry.get("lora", "")).exists()


def _next_action(entry):
    if not entry.get("design_prompt"):
        return "write a design_prompt (design_stage.py add ...)"
    if not _has_ref(entry):
        return "lock the still (gen-ref) or drop it at reference_image"
    if not _has_lora(entry):
        return "train the LoRA (see train-cmd)"
    return "READY -> sync-shots, then render_film"


def cmd_status(args):
    bible = load_bible(args.file)
    chars = bible.get("characters", {})
    if not chars:
        print("[design] no characters yet - add one with: design_stage.py add <id> ...")
        return
    print(f"{'ID':<12}{'TIER':<8}{'PROMPT':<8}{'STILL':<8}{'LORA':<8}NEXT ACTION")
    print("-" * 88)
    for cid, e in chars.items():
        print(f"{cid:<12}{e.get('tier','lead'):<8}"
              f"{('yes' if e.get('design_prompt') else 'no'):<8}"
              f"{('yes' if _has_ref(e) else 'no'):<8}"
              f"{('yes' if _has_lora(e) else 'no'):<8}"
              f"{_next_action(e)}")
    print(f"\nloras dir: {_loras_dir()}   (set COMFY_DIR to point elsewhere)")


def _t2i_graph(ckpt, prompt, negative, width, height, seed, prefix,
               steps=30, cfg=6.5):
    """Canonical SDXL text-to-image API graph (widely compatible in ComfyUI).
    FLUX users: swap CheckpointLoaderSimple for the FLUX unet/dualclip loaders."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}, "_meta": {"title": "ckpt"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]},
              "_meta": {"title": "POSITIVE_PROMPT"}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]},
              "_meta": {"title": "NEGATIVE_PROMPT"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


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


def cmd_gen_ref(args):
    bible = load_bible(args.file)
    entry = bible.get("characters", {}).get(args.id)
    if not entry:
        sys.exit(f"[design] no character '{args.id}' in {args.file}")
    if not entry.get("design_prompt"):
        sys.exit(f"[design] character '{args.id}' has no design_prompt yet")
    ckpt = args.ckpt or bible.get("image_model", "sd_xl_base_1.0.safetensors")
    w = args.width or bible.get("ref_width", 832)
    h = args.height or bible.get("ref_height", 1216)
    seed = args.seed if args.seed is not None else (entry.get("seed") or
                                                    int(time.time() * 1000) % (2 ** 31))
    prefix = f"ffchar_{args.id}"
    graph = _t2i_graph(ckpt, entry["design_prompt"], entry.get("negative", ""),
                       w, h, seed, prefix)
    print(f"[design] gen-ref '{args.id}' via {args.comfy} ckpt={ckpt} {w}x{h} seed={seed}")
    resp = _api(args.comfy, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    pid = resp["prompt_id"]
    out = None
    while True:
        time.sleep(2)
        hist = _api(args.comfy, f"/history/{pid}")
        if pid not in hist:
            continue
        for node_out in hist[pid].get("outputs", {}).values():
            for item in node_out.get("images", []) or []:
                if item.get("filename", "").lower().endswith((".png", ".jpg", ".jpeg")):
                    out = item
        st = hist[pid].get("status", {})
        if out or st.get("completed") or st.get("status_str") == "error":
            break
    if not out:
        sys.exit(f"[design] ComfyUI returned no image for '{args.id}'")
    dest = pathlib.Path(entry["reference_image"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"filename": out["filename"],
                                "subfolder": out.get("subfolder", ""),
                                "type": out.get("type", "output")})
    with urllib.request.urlopen(args.comfy.rstrip("/") + "/view?" + q, timeout=300) as r:
        dest.write_bytes(r.read())
    entry["seed"] = seed
    save_bible(args.file, bible)
    print(f"[design] saved canonical still -> {dest}  (seed {seed} recorded)")


def cmd_sync_shots(args):
    bible = load_bible(args.file)
    chars = bible.get("characters", {})
    shots_path = pathlib.Path(args.shots)
    if not shots_path.exists():
        sys.exit(f"[design] {args.shots} not found")
    spec = json.loads(shots_path.read_text())
    spec["characters"] = {cid: e["lora"] for cid, e in chars.items()}
    shots_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[design] synced {len(chars)} character(s) into {args.shots}: "
          f"{spec['characters']}")


def cmd_train_cmd(args):
    bible = load_bible(args.file)
    entry = bible.get("characters", {}).get(args.id)
    if not entry:
        sys.exit(f"[design] no character '{args.id}' in {args.file}")
    dataset = f"datasets/{args.id}"
    lora = entry.get("lora", f"{args.id}_v1.safetensors")
    print(f"# 1) Put 10-20 varied stills of '{args.id}' (built from "
          f"{entry.get('reference_image')}) into {dataset}/")
    print(f"# 2) Train (AI-Toolkit; disable the audio module to avoid OOM):")
    print(f"ai-toolkit train \\")
    print(f"  --dataset {dataset} \\")
    print(f"  --output {_loras_dir() / lora} \\")
    print(f"  --rank 32 --alpha 16 --lr 2e-4 --resolution 768 \\")
    print(f"  --trigger {args.id}")
    print(f"# 3) design_stage.py sync-shots  &&  render_film.py --shots shots.json")


def main():
    ap = argparse.ArgumentParser(description="FilmForge design stage (design->train->render)")
    ap.add_argument("--file", default=DEFAULT_FILE, help="the design bible (characters.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init"); sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add"); sp.add_argument("id")
    sp.add_argument("--name"); sp.add_argument("--prompt"); sp.add_argument("--negative")
    sp.add_argument("--tier", choices=["lead", "support", "extra"])
    sp.add_argument("--lora"); sp.add_argument("--strength", type=float)
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("status"); sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("gen-ref"); sp.add_argument("id")
    sp.add_argument("--comfy", default="http://127.0.0.1:8188")
    sp.add_argument("--ckpt"); sp.add_argument("--width", type=int)
    sp.add_argument("--height", type=int); sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_gen_ref)

    sp = sub.add_parser("sync-shots"); sp.add_argument("--shots", default="shots.json")
    sp.set_defaults(func=cmd_sync_shots)

    sp = sub.add_parser("train-cmd"); sp.add_argument("id")
    sp.set_defaults(func=cmd_train_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
