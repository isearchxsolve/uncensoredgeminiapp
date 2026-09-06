#!/usr/bin/env python3
"""
FilmForge backend - headless ComfyUI 'generate' endpoint on Modal.

ComfyUI runs IN THE CLOUD here - you do NOT need it installed locally.
This image installs ComfyUI, clones the custom-node packs, downloads the
models into a persistent Volume, and bakes in the tagged workflow.

Deploy ONCE PER MODEL so the router's lookup names match:
    FF_MODEL=ltx23 modal deploy modal_backend.py    # -> app 'ltx23-comfyui'
    FF_MODEL=wan22 modal deploy modal_backend.py    # -> app 'wan22-comfyui'

Each deploy exposes a warm class method generate(payload) that:
  - loads a pre-exported ComfyUI API-format workflow for that model
  - injects prompt / seconds / (optional) image by node TITLE
  - runs it headless and returns the finished MP4 as bytes

Validate the workflow node types against the deployed cloud ComfyUI (no local
install needed):
    FF_MODEL=ltx23 modal run modal_backend.py::verify_workflow
    FF_MODEL=wan22 modal run modal_backend.py::verify_workflow
"""
import base64, json, os, subprocess, time, urllib.request, uuid
from pathlib import Path
import modal

MODEL = os.environ.get("FF_MODEL", "ltx23")
assert MODEL in ("wan22", "ltx23"), "set FF_MODEL=wan22 or ltx23"
FPS = 24

MODELS_DIR = "/root/comfy/ComfyUI/models"
OUTPUT_DIR = "/root/comfy/ComfyUI/output"
INPUT_DIR = "/root/comfy/ComfyUI/input"
CUSTOM_NODES_DIR = "/root/comfy/ComfyUI/custom_nodes"
WORKFLOW_PATH = f"/root/{MODEL}_api.json"

# -------------------------------------------------------------------------
# Model file registry:  (huggingface_repo, filename_in_repo, comfy_subdir)
# -------------------------------------------------------------------------
MODEL_FILES = {
    "ltx23": [
        ("unsloth/LTX-2.3-GGUF", "ltx-2.3-22b-dev-Q4_K_M.gguf", "unet"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", "text_encoders"),
        ("unsloth/LTX-2.3-GGUF", "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors", "text_encoders"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_video_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_audio_vae.safetensors", "vae"),
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors", "loras"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "latent_upscale_models"),
    ],
    # Verified against live Hugging Face repos (Jul 2026). Wan 2.2 T2V 14B, GGUF Q4_K_M.
    # NOTE: the 14B GGUF quants live in the QuantStack repo, inside HighNoise/ and
    # LowNoise/ subfolders - you need BOTH experts (it's a Mixture-of-Experts model).
    # Text encoder + VAE come from the Comfy-Org Wan 2.2 repackaged repo; the 14B
    # model reuses the 2.1 VAE. (5B TI2V would use wan2.2_vae.safetensors instead.)
    "wan22": [
        ("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", "unet"),
        ("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf", "unet"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    ],
}

CUSTOM_NODES = {
    "ltx23": [
        "https://github.com/ltdrdata/ComfyUI-Manager",
        "https://github.com/city96/ComfyUI-GGUF",
        "https://github.com/pollockjj/ComfyUI-MultiGPU",
        "https://github.com/Lightricks/ComfyUI-LTXVideo",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "https://github.com/kijai/ComfyUI-KJNodes",
    ],
    "wan22": [
        "https://github.com/ltdrdata/ComfyUI-Manager",
        "https://github.com/city96/ComfyUI-GGUF",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "https://github.com/kijai/ComfyUI-KJNodes",
    ],
}

# -------------------------------------------------------------------------
# Image build
# -------------------------------------------------------------------------
def _build_image():
    img = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "wget", "libgl1", "libglib2.0-0")
        .pip_install("comfy-cli", "huggingface_hub[hf_transfer]", "requests")
        .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )
    for repo in CUSTOM_NODES[MODEL]:
        name = repo.rstrip("/").split("/")[-1]
        img = img.run_commands(
            f"cd {CUSTOM_NODES_DIR} && git clone --depth 1 {repo} {name} && "
            f"(pip install -r {name}/requirements.txt || true)"
        )
    # bake the pre-exported, node-tagged workflow into the image
    img = img.add_local_file(f"workflows/{MODEL}_api.json", WORKFLOW_PATH, copy=True)
    return img

image = _build_image()
vol = modal.Volume.from_name(f"{MODEL}-models", create_if_missing=True)
app = modal.App(f"{MODEL}-comfyui")

# -------------------------------------------------------------------------
# One-time model download into the persistent Volume
#   FF_MODEL=ltx23 modal run modal_backend.py::download_models
# -------------------------------------------------------------------------
@app.function(image=image, volumes={MODELS_DIR: vol}, timeout=60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def download_models():
    import shutil
    from huggingface_hub import hf_hub_download
    for repo, fname, subdir in MODEL_FILES[MODEL]:
        dest_dir = Path(MODELS_DIR) / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(repo_id=repo, filename=fname,
                                token=os.environ["HF_TOKEN"])
        target = dest_dir / Path(fname).name
        if not target.exists():
            shutil.copy(local, target)
        print("ok", target)
    vol.commit()
    print("all models present for", MODEL)

# -------------------------------------------------------------------------
# Warm headless ComfyUI - the router calls ComfyUI().generate.remote(payload)
# -------------------------------------------------------------------------
@app.cls(image=image, gpu=os.environ.get("FF_GPU", "L40S"),
         volumes={MODELS_DIR: vol}, timeout=60 * 60,
         scaledown_window=300,   # keep the container warm 5 min between jobs
         secrets=[modal.Secret.from_name("huggingface")])
class ComfyUI:
    @modal.enter()
    def start(self):
        subprocess.Popen(
            "comfy launch --background -- --listen 127.0.0.1 --port 8188",
            shell=True)
        for _ in range(180):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
                print("ComfyUI is up")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI did not start in time")

    @modal.method()
    def generate(self, payload: dict) -> bytes:
        prompt = payload["prompt"]
        seconds = float(payload.get("seconds", 5))
        image_b64 = payload.get("image")
        wf = json.loads(Path(WORKFLOW_PATH).read_text())
        wf = _inject(wf, prompt, seconds, image_b64)
        return _run_and_fetch(wf)

    @modal.method()
    def verify(self) -> dict:
        """Check the baked workflow's node types against this ComfyUI's
        installed nodes. No local ComfyUI needed - runs in the cloud."""
        wf = json.loads(Path(WORKFLOW_PATH).read_text())
        info = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8188/object_info", timeout=60).read())
        known = set(info.keys())
        used = sorted({n["class_type"] for n in wf.values()})
        missing = [c for c in used if c not in known]
        print("model", MODEL, "| nodes used:", used)
        return {"model": MODEL, "ok": not missing, "missing": missing}

# -------------------------------------------------------------------------
# Workflow injection - patch nodes by their title (see prep notes)
# -------------------------------------------------------------------------
def _inject(wf, prompt, seconds, image_b64):
    frames = max(1, round(seconds * FPS))
    for _id, node in wf.items():
        title = node.get("_meta", {}).get("title", "")
        if title == "POSITIVE_PROMPT":
            node["inputs"]["text"] = prompt
        elif title == "LENGTH":
            for k in ("length", "num_frames", "frames", "batch_size"):
                if k in node["inputs"]:
                    node["inputs"][k] = frames
        elif title == "SEED":
            for k in ("seed", "noise_seed"):
                if k in node["inputs"]:
                    node["inputs"][k] = uuid.uuid4().int % (2**32)
        elif title == "INPUT_IMAGE" and image_b64:
            Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
            img_name = f"input_{uuid.uuid4().hex}.png"
            (Path(INPUT_DIR) / img_name).write_bytes(base64.b64decode(image_b64))
            node["inputs"]["image"] = img_name
    return wf

def _run_and_fetch(wf) -> bytes:
    client_id = uuid.uuid4().hex
    body = json.dumps({"prompt": wf, "client_id": client_id}).encode()
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    prompt_id = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    # poll history until the job is done
    while True:
        time.sleep(2)
        hist = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10).read())
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            break
    return _newest_media(OUTPUT_DIR).read_bytes()

def _newest_media(d):
    exts = (".mp4", ".webm", ".mov", ".gif")
    files = [p for p in Path(d).rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise RuntimeError("no video output produced - check the workflow")
    return max(files, key=lambda p: p.stat().st_mtime)

# -------------------------------------------------------------------------
# Local test:  FF_MODEL=ltx23 modal run modal_backend.py
# -------------------------------------------------------------------------
@app.local_entrypoint()
def main(prompt: str = "a cinematic sunset over the ocean, waves rolling in",
         seconds: float = 5):
    out = ComfyUI().generate.remote({"prompt": prompt, "seconds": seconds})
    Path(f"{MODEL}_test.mp4").write_bytes(out)
    print("wrote", f"{MODEL}_test.mp4")

# -------------------------------------------------------------------------
# Validate workflow node types against the deployed cloud ComfyUI:
#   FF_MODEL=ltx23 modal run modal_backend.py::verify_workflow
# -------------------------------------------------------------------------
@app.local_entrypoint()
def verify_workflow():
    res = ComfyUI().verify.remote()
    if res["ok"]:
        print(f"[verify] {res['model']}: all workflow node types exist on the backend. OK")
    else:
        print(f"[verify] {res['model']}: MISSING node types -> {res['missing']}")
        print(f"  Fix workflows/{res['model']}_api.json (or install the node pack), then redeploy.")
