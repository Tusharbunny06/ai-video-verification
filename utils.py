import imageio
import torch
import numpy as np
from torchvision import transforms

def load_video(path, max_frames=32):
    reader = imageio.get_reader(path, "ffmpeg")

    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)

    reader.close()

    if len(frames) == 0:
        raise ValueError("No frames could be read from the video.")

    video = torch.stack(frames)  # (T, C, H, W)
    return video
