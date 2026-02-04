import torch
import imageio
import numpy as np
import hashlib
from PIL import Image
from diffusers import StableDiffusionPipeline

# -----------------------------
# LOAD TEXT → IMAGE MODEL
# -----------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

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
# IMAGE → VIDEO ANIMATION
# -----------------------------
def animate_image(image, num_frames=16):
    frames = []
    w, h = image.size

    for i in range(num_frames):
        scale = 1.0 + i * 0.01
        new_w, new_h = int(w * scale), int(h * scale)

        resized = image.resize((new_w, new_h))
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        cropped = resized.crop((left, top, left + w, top + h))

        frames.append(np.array(cropped))

    return frames

# -----------------------------
# PROMPT → VIDEO GENERATION
# -----------------------------
def generate_video_from_prompt(prompt, watermark_registry):
    # 1️⃣ Generate image from prompt
    image = pipe(prompt).images[0]

    # 2️⃣ Animate image into frames
    frames = animate_image(image)

    # 3️⃣ Save video
    output_path = "genai_prompt_video.mp4"
    imageio.mimsave(output_path, frames, fps=4)

    # 4️⃣ Register provenance
    video_hash = compute_hash(output_path)
    watermark_registry["prompt_video"] = {
        "hash": video_hash,
        "prompt": prompt,
        "generator": "Stable Diffusion + Animation (GenAI)"
    }

    return output_path