# FilmForge — zero-to-film pipeline (Wan 2.2 + LTX-2.3)

One folder, one script. Routes each shot to the best open model, fails over
between GPU providers, and stitches the clips into a finished film.

```
filmforge.py            the orchestrator: model choice + failover + stitching
modal_backend.py        the GPU backend you deploy to Modal (once per model)
filmforge_windows.bat    single Windows script: setup + run
setup.sh                 same, for macOS / Linux
shots.json               your shot list (edit this to make YOUR film)
workflows/               put your two exported ComfyUI workflows here
```

---

## The absolute fastest path

1. Keep every file above in ONE folder.
2. Install Python + ffmpeg once:
   - `winget install -e --id Python.Python.3.11`
   - `winget install -e --id Gyan.FFmpeg`
3. Double-click **filmforge_windows.bat**.
   - It builds a venv, installs deps, and runs an OFFLINE dry-run that produces
     `film_dryrun.mp4` (coloured placeholders + real crossfade + audio track).
     This proves routing + failover + stitching with NO account and NO GPU.
4. To make real video, do the two manual things below, then run the .bat again.

---

## Credentials the .bat needs (that's all)

| What | How you provide it | Notes |
|------|--------------------|-------|
| **Modal account** | `modal setup` opens your browser the first time | No key to paste — it's a browser login. Free tier is fine. |
| **Hugging Face token** | The .bat prompts you to paste it; it's stored as the Modal secret `huggingface` | Use a **read** token. REGENERATE the one you pasted in screenshots earlier. |
| **Beam token (optional)** | `BEAM_ENDPOINT` + `BEAM_TOKEN` — uncomment the two lines near the end of the .bat | Only needed if you want a 2nd free-credit provider for failover. |

No OpenAI / Google / Kaggle keys are needed anywhere.

---

## Where do the workflows come from?

The backend runs a ComfyUI graph in "API format." You don't hand-write it — you
get a working graph from ComfyUI, tag a few nodes, and export it. Sources:

**Wan 2.2**
- In ComfyUI: **Workflow → Browse Templates → Video → Wan 2.2** (built-in).
- Official examples: https://comfyanonymous.github.io/ComfyUI_examples/wan22/
- Docs: https://docs.comfy.org/tutorials/video/wan/wan2_2

**LTX-2.3**
- In ComfyUI: **Workflow → Browse Templates → Video → LTX**.
- The custom node repo ships examples: `ComfyUI-LTXVideo/example_workflows/`
  (https://github.com/Lightricks/ComfyUI-LTXVideo)

### Turn a template into a FilmForge workflow (once per model)
1. Open the template in ComfyUI and confirm it renders a clip.
2. Rename these node **titles** (right-click node → Title):
   - positive-prompt `CLIP Text Encode` → **POSITIVE_PROMPT**
   - the latent / length node (holds frame count) → **LENGTH**
   - the sampler / seed node → **SEED**
   - (image-to-video only) the `Load Image` node → **INPUT_IMAGE**
3. **Settings → enable Dev Mode Options**, then **Save (API Format)**.
4. Save them here as:
   - `workflows/ltx23_api.json`
   - `workflows/wan22_api.json`

The backend injects your prompt / length / seed / image by finding those titles.
Untagged nodes are left untouched.

---

## Wan 2.2 model files (verified Jul 2026)

Already set correctly in `modal_backend.py` — listed here so you can sanity-check:
- `QuantStack/Wan2.2-T2V-A14B-GGUF` -> `HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf`
- `QuantStack/Wan2.2-T2V-A14B-GGUF` -> `LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf`
- `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` -> `split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` -> `split_files/vae/wan_2.1_vae.safetensors`

Wan 2.2 is a Mixture-of-Experts model, so you need BOTH the high-noise and
low-noise GGUF files. Whatever loader nodes your exported `wan22_api.json` uses
must reference these exact filenames.

---

## Manual commands (if you'd rather not use the .bat)

```bash
pip install modal comfy-cli requests
modal setup
modal secret create huggingface HF_TOKEN=hf_your_NEW_token

# one-time weight download into each Modal Volume
FF_MODEL=ltx23 modal run  modal_backend.py::download_models
FF_MODEL=wan22 modal run  modal_backend.py::download_models

# deploy both warm backends (Wan wants a bigger GPU)
FF_MODEL=ltx23 modal deploy modal_backend.py
FF_MODEL=wan22 FF_GPU=A100-40GB modal deploy modal_backend.py

# make the film
python filmforge.py --shots shots.json --out film.mp4 --transition crossfade
# offline test any time:
python filmforge.py --shots shots.json --dry-run
# usage ledger:
python filmforge.py --status
```

---

## Security reminder
Regenerate the Hugging Face and Kaggle tokens that appeared in earlier
screenshots before using them here. Never commit tokens into these files.
