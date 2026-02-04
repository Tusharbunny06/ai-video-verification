import torch
import imageio
import numpy as np
import hashlib
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

# -----------------------------
# LOAD TEXT → IMAGE MODEL (SAFE)
# -----------------------------
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=scheduler,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")
pipe.enable_attention_slicing()  # reduces memory issues

# -----------------------------
# HASH UTILITY
# -----------------------------
def compute_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# -----------------------------
# IMAGE → VIDEO (VISIBLE MOTION)
# -----------------------------
def animate_image(image, num_frames=16):
    frames = []
    w, h = image.size

    for i in range(num_frames):
        shift = int(i * 6)
        canvas = Image.new("RGB", (w + shift, h))
        canvas.paste(image, (shift, 0))
        frame = canvas.crop((shift, 0, shift + w, h))
        frames.append(np.array(frame))

    return frames

# -----------------------------
# PROMPT → VIDEO GENERATION
# -----------------------------
def generate_video_from_prompt(prompt, watermark_registry):
    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=20,   # SAFE VALUE
            guidance_scale=7.5
        )

    image = result.images[0]

    # Animate image
    frames = animate_image(image)

    # Save video
    output_path = "genai_prompt_video.mp4"
    imageio.mimsave(output_path, frames, fps=4)

    # Register provenance
    video_hash = compute_hash(output_path)
    watermark_registry["prompt_video"] = {
        "hash": video_hash,
        "prompt": prompt,
        "generator": "Stable Diffusion + DDIM (GenAI)"
    }

    return output_path