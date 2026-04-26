import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_multispectral_resnet():
    """
    Modify a standard ResNet-18 to accept 4-channel input (RGB + NIR)
    and output a single continuous value for regression.
    """
    # Load pre-trained ResNet-18 weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 1. Modify the first convolutional layer
    # Original: conv1 (64 filters, 7x7 kernel, stride 2, padding 3, bias=False)
    # Current input_channels = 3 (RGB). New input_channels = 4 (RGB + NIR).
    original_conv = model.conv1
    
    model.conv1 = nn.Conv2d(
        in_channels=4, 
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )

    # Copy weights for RGB channels and initialize NIR channel
    with torch.no_grad():
        # Keep original RGB weights
        model.conv1.weight[:, :3, :, :] = original_conv.weight
        # Initialize NIR weights with the average of RGB weights (common practice)
        model.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

    # 2. Modify the Fully Connected (FC) layer for Regression
    # ResNet-18 output is 512 features before the classification layer
    num_ftrs = model.fc.in_features
    
    # Replace classification layer (1000 classes) with a single output neuron
    model.fc = nn.Linear(num_ftrs, 1)

    return model

if __name__ == "__main__":
    # Quick architecture validation
    net = get_multispectral_resnet()
    # Mock input: [batch_size, channels, height, width]
    mock_input = torch.randn(1, 4, 256, 256)
    output = net(mock_input)
    print(f"Input shape: {mock_input.shape}")
    print(f"Output shape: {output.shape} (1 means successful regression setup)")