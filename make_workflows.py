#!/usr/bin/env python3
"""
make_workflows.py - generate tagged, API-format ComfyUI workflows for the
FilmForge backend so you don't have to build + export them by hand.

Writes:
    workflows/ltx23_api.json     LTX-2.3 text-to-video
    workflows/wan22_api.json     Wan 2.2 T2V (Mixture-of-Experts: high+low)

Tagged with the titles the backend injects into:
    POSITIVE_PROMPT, NEGATIVE_PROMPT, LENGTH, SEED, CHARACTER_LORA

--------------------------------------------------------------------------
PER-CHARACTER LoRA  (consistency across stitched clips)
--------------------------------------------------------------------------
The #1 thing that breaks a multi-shot film is the hero's FACE drifting from
cut to cut. The fix is a per-character LoRA: train one small LoRA per
character, drop it in  ComfyUI/models/loras/<file>.safetensors , and render
EVERY shot of that character with the SAME LoRA. Then the stitched film keeps
one consistent identity.

  * ltx23_workflow(char_lora=..)  inserts ONE LoraLoaderModelOnly after the
    distilled speed-LoRA (LTX is single-model).
  * wan22_workflow(char_lora=..)  inserts TWO - one on the HIGH-noise expert
    and one on the LOW-noise expert - because Wan 2.2 is a two-expert model
    and a LoRA on only one half produces flicker.

render_film.py builds a fresh graph per shot with that shot's character LoRA,
so you never hand-edit JSON. These CLI-generated files are the no-character
baseline you can import into the ComfyUI UI.

--------------------------------------------------------------------------
NODE NAMES VALIDATED AGAINST THE INTERNET (Jul 2026)
--------------------------------------------------------------------------
Wan 2.2 (built-in + ComfyUI-GGUF + VideoHelperSuite):
  UnetLoaderGGUF . LoraLoaderModelOnly . CLIPLoader(type="wan") .
  CLIPTextEncode . VAELoader . ModelSamplingSD3 .
  EmptyHunyuanLatentVideo . KSamplerAdvanced x2 (high->low) . VAEDecode .
  VHS_VideoCombine.
LTX-2.3: LTXVGemmaCLIPModelLoader(gemma_path, ltxv_path, max_length -> CLIP).
  Its graph evolves fast; for the UI, prefer Lightricks' OFFICIAL workflow:
      python make_workflows.py --fetch-official
  Validate node types against YOUR install before the first paid render:
      python make_workflows.py --verify http://127.0.0.1:8188

Existing files are NOT overwritten unless you pass --force.
Model filenames below are kept in sync with modal_backend.py MODEL_FILES.
"""
import argparse, json, os, urllib.request

OUT_DIR = "workflows"
FPS = 24

# --- filenames (must match modal_backend.py MODEL_FILES) --------------------
FF_QUANT      = os.environ.get("FF_QUANT", "q4_k_m").lower()
_QUANT_TAG    = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q6_k": "Q6_K"}
LTX_UNET      = f"ltx-2.3-22b-dev-{_QUANT_TAG.get(FF_QUANT, 'Q4_K_M')}.gguf"
LTX_TEXT_GGUF = "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
LTX_VIDEO_VAE = "ltx-2.3-22b-dev_video_vae.safetensors"
LTX_LORA      = "ltx-2.3-22b-distilled-lora-384.safetensors"

WAN_HIGH = "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
WAN_LOW  = "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
WAN_UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE  = "wan_2.1_vae.safetensors"

NEG = "blurry, distorted, low quality, watermark, text"
DEFAULT_PROMPT = "a cinematic establishing shot, natural light, film grain"

# Default character-LoRA strength (0 = off, ~0.7-0.9 = strong identity lock).
CHAR_STRENGTH = 0.85

# Optional 4x upscaler. Drop the .pth in ComfyUI/models/upscale_models/.
# lightning_download.py fetches 4x-UltraSharp by default (best-effort).
UPSCALE_MODEL = "4x-UltraSharp.pth"

# --- Golden Quality env knobs (see the Max-Quality convergence page) --------
# ALL optional. With nothing set, the defaults below reproduce the fast
# distilled preset EXACTLY, so an unset environment renders identically to
# before (loop-safe). preflight.py builds its graphs from THESE values, so a
# live --comfy check validates whatever you dial in BEFORE any GPU time is
# spent - a missing quant file or unknown node is caught at the gate, not
# after you have paid for the render.
def _envi(name, d):
    try:
        return int(os.environ.get(name, d))
    except Exception:
        return d


def _envf(name, d):
    try:
        return float(os.environ.get(name, d))
    except Exception:
        return d


# FF_DISTILLED=1 (default) keeps the few-step speed LoRA; =0 is full-dev.
FF_DISTILLED = os.environ.get("FF_DISTILLED", "1") != "0"
# Steps/CFG default to the correct preset for the chosen mode.
FF_STEPS     = _envi("FF_STEPS", 8 if FF_DISTILLED else 28)
FF_CFG       = _envf("FF_CFG", 1.0 if FF_DISTILLED else 4.0)
FF_SHIFT     = _envf("FF_SHIFT", 0.0)          # >0 inserts a ModelSamplingSD3 sigma-shift on LTX
FF_SAMPLER   = os.environ.get("FF_SAMPLER", "euler")
FF_SCHEDULER = os.environ.get("FF_SCHEDULER", "simple")
FF_DRAFT_RES = _envi("FF_DRAFT_RES", 0)        # >0 shrinks BASE height (draft-then-upscale); 0 = native
FF_TILED_VAE = os.environ.get("FF_TILED_VAE", "0") == "1"  # VAEDecodeTiled instead of VAEDecode

# --- manifest override ------------------------------------------------------
# lightning_download.py writes models/filmforge_manifest.json mapping a stable
# ROLE to the ACTUAL basename it placed on disk. Applying it here makes every
# generated graph reference exactly what was downloaded, which eliminates the
# glob-vs-hardcode filename mismatch class (e.g. Gemma Q4_K_M vs Q4_K_XL). If no
# manifest exists, the hardcoded defaults above are used unchanged.
_ROLE_TO_GLOBAL = {
    "ltx_unet": "LTX_UNET",
    "ltx_text_gguf": "LTX_TEXT_GGUF",
    "ltx_video_vae": "LTX_VIDEO_VAE",
    "ltx_lora": "LTX_LORA",
    "wan_high": "WAN_HIGH",
    "wan_low": "WAN_LOW",
    "wan_umt5": "WAN_UMT5",
    "wan_vae": "WAN_VAE",
    "upscale": "UPSCALE_MODEL",
}


def apply_manifest(path=None):
    """Override filename constants from models/filmforge_manifest.json if present.
    Returns {global_name: filename} actually applied. Safe no-op if missing."""
    if path is None:
        comfy = os.environ.get("COMFY_DIR", os.path.expanduser("~/ComfyUI"))
        path = os.path.join(comfy, "models", "filmforge_manifest.json")
    applied = {}
    try:
        with open(path) as f:
            man = json.load(f)
    except Exception:
        return applied
    g = globals()
    for role, gname in _ROLE_TO_GLOBAL.items():
        entry = man.get(role)
        fname = entry.get("file") if isinstance(entry, dict) else entry
        if fname:
            g[gname] = fname
            applied[gname] = fname
    return applied

# Maintained upstream workflows (import into ComfyUI, then Save in API Format).
OFFICIAL_WORKFLOWS = {
    "official_ltx23_T2V_I2V.json":
        "https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/"
        "example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json",
}
OFFICIAL_WAN_DOCS = "https://comfyanonymous.github.io/ComfyUI_examples/wan22/"


def ltx23_workflow(width=768, height=448, seconds=5,
                   char_lora=None, char_strength=CHAR_STRENGTH, upscale=False):
    """LTX-2.3 T2V graph. Honors the Golden Quality env knobs (FF_QUANT via
    LTX_UNET, FF_DISTILLED, FF_STEPS, FF_CFG, FF_SHIFT, FF_SAMPLER,
    FF_SCHEDULER, FF_DRAFT_RES, FF_TILED_VAE). Defaults reproduce the fast
    distilled preset exactly. If char_lora is given, a character LoRA is
    stacked so this shot renders the SAME identity as every other shot."""
    length = int(round(seconds * FPS)) // 8 * 8 + 1   # LTX wants 8k+1 frames
    # draft-then-upscale: shrink the BASE render to lock motion cheaply, then
    # let the upscale stage add detail. Snap to /32 (LTX dimension rule).
    if FF_DRAFT_RES and FF_DRAFT_RES < height:
        width = max(32, int(round(width * FF_DRAFT_RES / height / 32)) * 32)
        height = max(32, int(round(FF_DRAFT_RES / 32)) * 32)
    g = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": LTX_UNET}},
        # VALIDATED node: LTXVGemmaCLIPModelLoader (gemma_path, ltxv_path, max_length -> CLIP)
        "3": {"class_type": "LTXVGemmaCLIPModelLoader",
              "inputs": {"gemma_path": LTX_TEXT_GGUF, "ltxv_path": LTX_UNET,
                         "max_length": 1024}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": DEFAULT_PROMPT, "clip": ["3", 0]},
              "_meta": {"title": "POSITIVE_PROMPT"}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["3", 0]},
              "_meta": {"title": "NEGATIVE_PROMPT"}},
        "6": {"class_type": "VAELoader",
              "inputs": {"vae_name": LTX_VIDEO_VAE}},
        "7": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": width, "height": height,
                         "length": length, "batch_size": 1},
              "_meta": {"title": "LENGTH"}},
        "8": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                         "frame_rate": FPS}},
    }
    # Speed vs quality: distilled keeps the few-step LoRA; full-dev drops it
    # and relies on FF_STEPS/FF_CFG for the real quality ceiling.
    if FF_DISTILLED:
        g["2"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": LTX_LORA,
                             "strength_model": 1.0}}
        base_src = ["2", 0]
    else:
        base_src = ["1", 0]              # full-dev: no speed LoRA
    model_src = base_src
    if char_lora:
        g["2c"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": base_src, "lora_name": char_lora,
                              "strength_model": float(char_strength)},
                   "_meta": {"title": "CHARACTER_LORA"}}
        model_src = ["2c", 0]
    # Optional sigma-shift (cheap quality lever). OFF by default so the default
    # node set is unchanged; run --verify before a paid render if you enable it.
    if FF_SHIFT > 0:
        g["8s"] = {"class_type": "ModelSamplingSD3",
                   "inputs": {"model": model_src, "shift": FF_SHIFT}}
        model_src = ["8s", 0]
    g["9"] = {"class_type": "KSampler",
              "inputs": {"model": model_src, "positive": ["8", 0],
                         "negative": ["8", 1], "latent_image": ["7", 0],
                         # distilled default: 8 steps + cfg 1.0. full-dev
                         # default: 28 steps + cfg 4.0 (set FF_DISTILLED=0).
                         "seed": 0, "steps": FF_STEPS, "cfg": FF_CFG,
                         "sampler_name": FF_SAMPLER, "scheduler": FF_SCHEDULER,
                         "denoise": 1.0},
              "_meta": {"title": "SEED"}}
    g["10"] = {"class_type": "VAEDecodeTiled" if FF_TILED_VAE else "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["6", 0]}}
    if FF_TILED_VAE:
        g["10"]["inputs"]["tile_size"] = 512   # lower this if you still OOM
    img_src = ["10", 0]
    if upscale:
        g["10u"] = {"class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": UPSCALE_MODEL}}
        g["10v"] = {"class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["10u", 0], "image": ["10", 0]},
                    "_meta": {"title": "UPSCALE_4X"}}
        img_src = ["10v", 0]
    g["11"] = {"class_type": "VHS_VideoCombine",
               "inputs": {"images": img_src, "frame_rate": FPS,
                          "format": "video/h264-mp4",
                          "filename_prefix": "ltx23"}}
    return g


def wan22_workflow(width=1280, height=720, seconds=5,
                   char_lora=None, char_strength=CHAR_STRENGTH, upscale=False):
    """Wan 2.2 T2V (two-expert). A character LoRA is stacked on BOTH the high-
    and low-noise experts (Wan 2.2 needs both, or the identity flickers)."""
    length = int(round(seconds * FPS)) // 4 * 4 + 1
    half = 10
    g = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": WAN_HIGH}},
        "2": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": WAN_LOW}},
    }
    high_src, low_src = ["1", 0], ["2", 0]
    if char_lora:
        g["1c"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": char_lora,
                              "strength_model": float(char_strength)},
                   "_meta": {"title": "CHARACTER_LORA"}}
        g["2c"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["2", 0], "lora_name": char_lora,
                              "strength_model": float(char_strength)},
                   "_meta": {"title": "CHARACTER_LORA_LOW"}}
        high_src, low_src = ["1c", 0], ["2c", 0]
    g["3"] = {"class_type": "ModelSamplingSD3",
              "inputs": {"model": high_src, "shift": 8.0}}
    g["4"] = {"class_type": "ModelSamplingSD3",
              "inputs": {"model": low_src, "shift": 8.0}}
    g["5"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_UMT5, "type": "wan"}}
    g["6"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": DEFAULT_PROMPT, "clip": ["5", 0]},
              "_meta": {"title": "POSITIVE_PROMPT"}}
    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["5", 0]},
              "_meta": {"title": "NEGATIVE_PROMPT"}}
    g["8"] = {"class_type": "VAELoader",
              "inputs": {"vae_name": WAN_VAE}}
    g["9"] = {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": width, "height": height,
                         "length": length, "batch_size": 1},
              "_meta": {"title": "LENGTH"}}
    g["10"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["3", 0], "positive": ["6", 0],
                          "negative": ["7", 0], "latent_image": ["9", 0],
                          "add_noise": "enable", "noise_seed": 0,
                          "steps": 20, "cfg": 3.5, "sampler_name": "euler",
                          "scheduler": "simple", "start_at_step": 0,
                          "end_at_step": half,
                          "return_with_leftover_noise": "enable"},
               "_meta": {"title": "SEED"}}
    g["11"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["4", 0], "positive": ["6", 0],
                          "negative": ["7", 0], "latent_image": ["10", 0],
                          "add_noise": "disable", "noise_seed": 0,
                          "steps": 20, "cfg": 3.5, "sampler_name": "euler",
                          "scheduler": "simple", "start_at_step": half,
                          "end_at_step": 10000,
                          "return_with_leftover_noise": "disable"}}
    g["12"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["8", 0]}}
    img_src = ["12", 0]
    if upscale:
        g["12u"] = {"class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": UPSCALE_MODEL}}
        g["12v"] = {"class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["12u", 0], "image": ["12", 0]},
                    "_meta": {"title": "UPSCALE_4X"}}
        img_src = ["12v", 0]
    g["13"] = {"class_type": "VHS_VideoCombine",
               "inputs": {"images": img_src, "frame_rate": FPS,
                          "format": "video/h264-mp4",
                          "filename_prefix": "wan22"}}
    return g


def build(model, width, height, seconds, char_lora=None,
          char_strength=CHAR_STRENGTH, upscale=False):
    """Dispatch helper used by render_film.py."""
    if model == "wan22":
        return wan22_workflow(width, height, seconds, char_lora, char_strength, upscale)
    return ltx23_workflow(width, height, seconds, char_lora, char_strength, upscale)


def fetch_official():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, url in OFFICIAL_WORKFLOWS.items():
        path = os.path.join(OUT_DIR, fname)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            json.loads(data)  # sanity check it's valid JSON
            with open(path, "wb") as f:
                f.write(data)
            print(f"[fetch] {path}  <-  {url}")
        except Exception as e:
            print(f"[fetch] FAILED {url}: {e}")
    print(f"[note] Wan official examples (download JSON manually): {OFFICIAL_WAN_DOCS}")
    print("[note] Official files are UI-format: open in ComfyUI, then")
    print("       Save (API Format) as workflows/ltx23_api.json to use with the backend.")


def verify(graphs, base_url):
    base_url = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(base_url + "/object_info", timeout=15) as r:
            known = set(json.loads(r.read()).keys())
    except Exception as e:
        print(f"[verify] could not reach ComfyUI at {base_url}: {e}")
        return False
    ok = True
    for name, wf in graphs.items():
        missing = sorted({n["class_type"] for n in wf.values()
                          if n["class_type"] not in known})
        if missing:
            ok = False
            print(f"[verify] {name}: MISSING node types -> {missing}")
            print("         install the matching custom-node pack or fix the class_type.")
        else:
            print(f"[verify] {name}: all node types exist on the server. OK")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing workflow files")
    ap.add_argument("--verify", metavar="COMFYUI_URL",
                    help="validate node types against a running ComfyUI, e.g. http://127.0.0.1:8188")
    ap.add_argument("--fetch-official", action="store_true",
                    help="download Lightricks' official LTX-2.3 workflow for reference")
    ap.add_argument("--character", metavar="LORA_FILE",
                    help="bake a character LoRA (in models/loras) into the generated files")
    ap.add_argument("--char-strength", type=float, default=CHAR_STRENGTH,
                    help=f"character LoRA strength (default {CHAR_STRENGTH})")
    ap.add_argument("--upscale", action="store_true",
                    help=f"append a 4x upscale stage (needs models/upscale_models/{UPSCALE_MODEL})")
    args = ap.parse_args()

    if args.fetch_official:
        fetch_official()
        return

    applied = apply_manifest()
    if applied:
        print(f"[manifest] using downloaded filenames: {applied}")
    print(f"[ltx-config] quant={FF_QUANT} unet={LTX_UNET} "
          f"mode={'distilled' if FF_DISTILLED else 'full-dev'} "
          f"steps={FF_STEPS} cfg={FF_CFG} shift={FF_SHIFT or 'off'} "
          f"sampler={FF_SAMPLER}/{FF_SCHEDULER} "
          f"draft_res={FF_DRAFT_RES or 'native'} "
          f"tiled_vae={'on' if FF_TILED_VAE else 'off'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    graphs = {
        "ltx23_api.json": ltx23_workflow(char_lora=args.character,
                                         char_strength=args.char_strength,
                                         upscale=args.upscale),
        "wan22_api.json": wan22_workflow(char_lora=args.character,
                                         char_strength=args.char_strength,
                                         upscale=args.upscale),
    }

    for fname, wf in graphs.items():
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path) and not args.force:
            print(f"[skip] {path} already exists (use --force to overwrite)")
            continue
        with open(path, "w") as f:
            json.dump(wf, f, indent=2)
        print(f"[write] {path}  ({len(wf)} nodes"
              + (", +character LoRA" if args.character else "") + ")")

    if args.verify:
        verify(graphs, args.verify)
    else:
        print("\nNext:")
        print("  validate against your ComfyUI:  python make_workflows.py --verify http://127.0.0.1:8188")
        print("  render a whole film w/ per-shot LoRAs:  python render_film.py --shots shots.json")


if __name__ == "__main__":
    main()
