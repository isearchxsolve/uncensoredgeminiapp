#!/usr/bin/env python3
"""
Download FilmForge model weights into a ComfyUI models dir on a Lightning Studio.

    export HF_TOKEN=hf_your_token
    python lightning_download.py ltx23     # or: wan22

Robust by design: each component tries several (repo, pattern) candidates and
downloads the first that matches, so minor filename drift on HuggingFace won't
break the run. Missing components are reported at the end instead of crashing.
Set COMFY_DIR if your ComfyUI is not at ~/ComfyUI.
"""
import os, sys, shutil, fnmatch, json
from pathlib import Path

# --- Reliability: force the classic, resumable HTTP downloader. -------------
# Lightning's newer "Xet" transfer can hang mid-file (the progress bar and the
# elapsed timer freeze instead of erroring). Disabling it gives a steady,
# always-moving, resumable download. MUST be set before importing huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Be patient with big files on shared networks before giving up.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    sys.exit("Run:  pip install 'huggingface_hub[hf_transfer]'  first")

COMFY = os.environ.get("COMFY_DIR", str(Path.home() / "ComfyUI"))
MODELS_DIR = Path(COMFY) / "models"

# Golden Quality: FF_QUANT picks which LTX UNet precision to fetch. Higher =
# more detail, more VRAM (q6_k ~18GB, q5_k_m ~15.5GB, q4_k_m ~13GB on 24GB L4).
# make_workflows.py reads the SAME env var so the graph references what landed.
FF_QUANT = os.environ.get("FF_QUANT", "q4_k_m").lower()
_QUANT_TAG = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q6_k": "Q6_K"}
_QTAG = _QUANT_TAG.get(FF_QUANT, "Q4_K_M")

# Each component = (role, target subdir, ordered list of (repo, pattern) candidates).
# pattern may be an exact filename or a glob (e.g. "*Q4_K_M*.gguf"). First hit wins.
# `role` is a STABLE key written to filmforge_manifest.json so the workflow
# builder can reference the EXACT filename that landed on disk (see make_workflows
# apply_manifest). This removes the glob-vs-hardcode filename mismatch class.
COMPONENTS = {
    "ltx23": [
        ("ltx_unet", "unet", [
            # requested precision first, then looser globs, then Q4 fallback so
            # a render is always possible even if the exact quant is absent.
            ("unsloth/LTX-2.3-GGUF", f"ltx-2.3-22b-dev-{_QTAG}.gguf"),
            ("unsloth/LTX-2.3-GGUF", f"*dev-{_QTAG}.gguf"),
            ("unsloth/LTX-2.3-GGUF", "ltx-2.3-22b-dev-Q4_K_M.gguf"),
        ]),
        ("ltx_text_gguf", "text_encoders", [   # Gemma-3 12B QAT text encoder (MUST be the qat repo)
            # exact name the workflow expects FIRST, then progressively looser globs
            ("unsloth/gemma-3-12b-it-qat-GGUF", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"),
            ("unsloth/gemma-3-12b-it-qat-GGUF", "*Q4_K_XL*.gguf"),
            ("unsloth/gemma-3-12b-it-qat-GGUF", "*Q4_K_M*.gguf"),
            ("unsloth/gemma-3-12b-it-qat-GGUF", "*Q4_K*.gguf"),
            ("unsloth/gemma-3-12b-it-qat-GGUF", "*IQ4*.gguf"),
        ]),
        ("ltx_connector", "text_encoders", [   # LTX embeddings connector (CLIP #2, type ltxv)
            ("unsloth/LTX-2.3-GGUF", "*embeddings_connector*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*embeddings_connector*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*connector*.safetensors"),
        ]),
        ("ltx_text_projection", "text_encoders", [   # text projection (Kijai split components)
            ("Kijai/LTX2.3_comfy", "*text_projection*.safetensors"),
        ]),
        ("ltx_video_vae", "vae", [             # video VAE
            ("unsloth/LTX-2.3-GGUF", "*video_vae*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*video*vae*.safetensors"),
            ("Kijai/LTX2.3_comfy", "vae/*video*.safetensors"),
        ]),
        ("ltx_audio_vae", "vae", [             # audio VAE (for synced audio; optional)
            ("unsloth/LTX-2.3-GGUF", "*audio_vae*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*audio*vae*.safetensors"),
        ]),
        ("ltx_lora", "loras", [           # distilled speed LoRA
            ("Lightricks/LTX-2.3", "*distilled-lora-384*.safetensors"),
        ]),
        # Native LTX upscalers (best-effort, OPTIONAL). Pre-staged on disk so
        # that when you import Lightricks' OFFICIAL draft-then-upscale workflow
        # (make_workflows.py --fetch-official) the files are already present.
        # These synthesize detail (spatial) and smooth fast motion (temporal)
        # far better than the generic 4x ESRGAN. If a glob misses, it is simply
        # reported as missing - never fatal.
        ("ltx_spatial_upscaler", "upscale_models", [
            ("Lightricks/LTX-2.3", "*spatial*upscaler*.safetensors"),
            ("Lightricks/LTX-Video", "*spatial*upscaler*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*spatial*upscal*.safetensors"),
        ]),
        ("ltx_temporal_upscaler", "upscale_models", [
            ("Lightricks/LTX-2.3", "*temporal*upscaler*.safetensors"),
            ("Lightricks/LTX-Video", "*temporal*upscaler*.safetensors"),
            ("Kijai/LTX2.3_comfy", "*temporal*upscal*.safetensors"),
        ]),
        ("upscale", "upscale_models", [  # optional 4x upscaler (best-effort; for >90% fidelity)
            ("Kim2091/UltraSharp", "4x-UltraSharp.pth"),
            ("uwg/upscaler", "ESRGAN/4x-UltraSharp.pth"),
        ]),
    ],
    "wan22": [
        ("wan_high", "unet", [("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")]),
        ("wan_low", "unet", [("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf")]),
        ("wan_umt5", "text_encoders", [("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")]),
        ("wan_vae", "vae", [("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors")]),
        ("upscale", "upscale_models", [  # optional 4x upscaler (best-effort; for >90% fidelity)
            ("Kim2091/UltraSharp", "4x-UltraSharp.pth"),
            ("uwg/upscaler", "ESRGAN/4x-UltraSharp.pth"),
        ]),
    ],
}

def has_glob(s):
    return any(c in s for c in "*?[")

def resolve(repo, pattern, token, cache):
    """Return repo file paths matching pattern (exact string or glob)."""
    if not has_glob(pattern):
        return [pattern]
    if repo not in cache:
        try:
            cache[repo] = HfApi().list_repo_files(repo_id=repo, token=token)
        except Exception as e:
            print(f"   (could not list {repo}: {e})")
            cache[repo] = []
    return [f for f in cache[repo] if fnmatch.fnmatch(f, pattern)]

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ltx23"
    if model not in COMPONENTS:
        sys.exit(f"model must be one of {list(COMPONENTS)}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set - gated repos may fail\n")
    print(f"(classic HTTP download; Xet disabled) -> {MODELS_DIR}\n")
    cache = {}
    got, missed = 0, []
    manifest = {}
    for role, subdir, candidates in COMPONENTS[model]:
        dest = MODELS_DIR / subdir
        dest.mkdir(parents=True, exist_ok=True)
        done = False
        for repo, pattern in candidates:
            for fname in resolve(repo, pattern, token, cache):
                try:
                    local = hf_hub_download(repo_id=repo, filename=fname, token=token)
                except Exception as e:
                    print(f"   skip {repo}/{fname}: {e}")
                    continue
                target = dest / Path(fname).name
                if not target.exists():
                    shutil.copy(local, target)
                print("ok ", target)
                got += 1
                done = True
                # record the FIRST file obtained for this role as its canonical
                # on-disk name so the workflow builder references THIS exact file
                manifest.setdefault(role, {"subdir": subdir, "file": target.name})
            if done:
                break
        if not done:
            missed.append(f"{role} ({subdir}) <- {candidates[0][0]} {candidates[0][1]}")
            print(f"!! no file found for {role}/{subdir} ({candidates[0][1]})")
    # Write/merge a manifest so make_workflows references ACTUAL on-disk filenames
    # (eliminates the download-glob vs workflow-hardcode mismatch class, e.g.
    # Gemma Q4_K_M vs Q4_K_XL). Merges so ltx23 + wan22 runs accumulate.
    man_path = MODELS_DIR / "filmforge_manifest.json"
    try:
        existing = {}
        if man_path.exists():
            existing = json.loads(man_path.read_text())
        existing.update(manifest)
        man_path.write_text(json.dumps(existing, indent=2))
        print(f"manifest -> {man_path} ({len(manifest)} role(s) for {model})")
    except Exception as e:
        print(f"   (could not write manifest: {e})")
    print(f"\ndone -> {model}: {got} file(s) into {MODELS_DIR}")
    if missed:
        print("MISSING (grab manually if a workflow needs it):")
        for m in missed:
            print("   -", m)

if __name__ == "__main__":
    main()
