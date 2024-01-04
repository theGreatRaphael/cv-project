import torch.nn as nn

from src.unet import UNet
from src.focus_stack import FocusStackingNetwork

class CVNetwork(nn.Module):
    def __init__(self) -> None:
        super(CVNetwork, self).__init__()

        self.focus_stack = FocusStackingNetwork()
        self.unet = UNet(n_channels=1, n_classes=1)  # 1 input channel, 1 output channel

    def forward(self, x):
        x = self.focus_stack(x)
        logits = self.unet(x)

        return logits
    

