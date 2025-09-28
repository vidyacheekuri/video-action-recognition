# =============================
# 3. MODEL DEFINITION (CNN + LSTM)
# =============================
import torch
import torch.nn as nn
from typing import Optional

try:
    # torchvision video models (PyTorch >= 0.13)
    from torchvision.models.video import r3d_18, R3D_18_Weights
    _HAS_TV_VIDEO = True
except Exception:
    _HAS_TV_VIDEO = False

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(64 * 28 * 28, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class R3D18Classifier(nn.Module):
    """
    Wrapper around torchvision r3d_18 with adjustable number of classes.
    Expects input tensor in shape (B, T, C, H, W).
    """

    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if not _HAS_TV_VIDEO:
            raise ImportError("torchvision.models.video.r3d_18 is not available in this environment")

        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.backbone = r3d_18(weights=weights)
            self.normalize_transform = weights.transforms()
        else:
            self.backbone = r3d_18(weights=None)
            self.normalize_transform = None

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision video models expect (B, C, T, H, W)
        if x.dim() != 5:
            raise ValueError(f"Expected input of shape (B, T, C, H, W), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)