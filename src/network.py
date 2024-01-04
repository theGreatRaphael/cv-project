import torch.nn as nn

from unet import UNet
from focus_stack import FocusStackingNetwork

class CVNetwork(nn.Module):
    def __init__(self, num_input_images) -> None:
        super(CVNetwork, self).__init__()

        self.focus_stack = FocusStackingNetwork(num_input_images=num_input_images)
        self.unet = UNet(n_channels=1, n_classes=1)  # 1 input channel, 1 output channel

    def forward(self, x):
        x = self.focus_stack(x)
        logits = self.unet(x)

        return logits
    

