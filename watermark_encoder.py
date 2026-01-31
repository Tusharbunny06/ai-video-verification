import torch
from watermark_signature import WATERMARK_BITS

def embed_watermark(video):
    """
    video: (T, 3, H, W)
    Embeds a simple scalar watermark signal
    """
    # Convert watermark bits into a scalar signal
    watermark_signal = WATERMARK_BITS.float().mean()

    strength = 0.08  # small but detectable

    watermarked = video + strength * watermark_signal
    watermarked = torch.clamp(watermarked, 0, 1)

    return watermarked
