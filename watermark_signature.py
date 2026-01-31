import torch

# Fixed watermark signature (private key)
WATERMARK_BITS = torch.randint(0, 2, (32,), dtype=torch.float32)
