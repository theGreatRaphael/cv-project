import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class FusionLayer(nn.Module):
    """A fusion layer that takes a list of feature maps and returns their element-wise maximum."""
    def forward(self, features):
        return torch.max(torch.stack(features), dim=0)[0]

class FocusStackingNetwork(nn.Module):
    """A focus stacking model that processes multiple input images to produce a single, all-in-focus output."""
    def __init__(self, num_input_images=4, num_channels=1):
        super(FocusStackingNetwork, self).__init__()
        self.num_input_images = num_input_images
        self.conv_blocks = nn.ModuleList([ConvBlock(num_channels, num_channels) for _ in range(num_input_images)])
        self.fusion_layer = FusionLayer()
        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        processed_images = [self.conv_blocks[i](inputs[i]) for i in range(self.num_input_images)]
        
        fused = self.fusion_layer(processed_images)
        
        output = self.final_conv(fused)
        return output
