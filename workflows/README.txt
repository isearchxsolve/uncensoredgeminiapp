Put your two exported ComfyUI workflows here:

    ltx23_api.json
    wan22_api.json

How to make them (once):
  1. Open ComfyUI -> Workflow -> Browse Templates -> Video -> pick LTX / Wan 2.2
  2. Confirm it generates a clip.
  3. Rename node TITLES (right-click node -> Title):
        positive prompt CLIPTextEncode  ->  POSITIVE_PROMPT
        latent/length node               ->  LENGTH
        sampler/seed node                ->  SEED
        Load Image node (i2v only)       ->  INPUT_IMAGE
  4. Settings -> enable Dev Mode Options
  5. Save (API Format)  ->  save into this folder with the names above.

Sources:
  Wan 2.2 : https://comfyanonymous.github.io/ComfyUI_examples/wan22/
  LTX     : https://github.com/Lightricks/ComfyUI-LTXVideo (example_workflows/)

The backend (modal_backend.py) injects your prompt / length / seed / image by
looking up those node titles. Untagged nodes are left as-is.
