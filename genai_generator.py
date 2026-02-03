import shutil
import hashlib
import os

def compute_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_ai_video(prompt, source_video, watermark_registry):
    """
    Simulates a Generative AI video model.
    Copies a sample video and registers it as AI-generated.
    """

    output_path = "generated_ai_video.mp4"

    # Simulate generation (copy as placeholder)
    shutil.copy(source_video, output_path)

    # Compute provenance hash
    video_hash = compute_hash(output_path)

    # Register provenance (watermark)
    watermark_registry["generated_ai_video"] = {
        "hash": video_hash,
        "prompt": prompt,
        "generator": "Simulated GenAI Video Model v1"
    }

    return output_path
