import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet-18
        backbone = models.resnet18(pretrained=True)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone (VERY IMPORTANT)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video):
        """
        video: (T, 3, H, W)
        """
        features = []

        for t in range(video.shape[0]):
            frame = video[t].unsqueeze(0)  # (1, 3, H, W)
            feat = self.feature_extractor(frame)
            feat = feat.view(-1)  # (512,)
            features.append(feat)

        # Temporal average
        video_feature = torch.stack(features).mean(dim=0)

        return self.classifier(video_feature)
