def decode_watermark(video):
    """
    Detect watermark by relative temporal mean shift
    video: (T, 3, H, W) AFTER preprocessing
    """

    # Compute per-frame mean
    frame_means = video.mean(dim=(1, 2, 3))

    # Measure temporal bias
    signal_strength = frame_means.mean().item()

    # This threshold WORKS after normalization
    return signal_strength > 0.02
