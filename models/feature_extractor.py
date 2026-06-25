import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block: Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block: Conv3d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet basic block for feature extraction"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) module from PSMNet.
    Uses adaptive average pooling at different scales to capture multi-scale context.
    Works with any input size.
    """
    def __init__(self, in_channels):
        super(SpatialPyramidPooling, self).__init__()
        # Use adaptive pooling with fixed output sizes
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            ConvBlock(in_channels, 32, kernel_size=1, padding=0),
        )
        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            ConvBlock(in_channels, 32, kernel_size=1, padding=0),
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            ConvBlock(in_channels, 32, kernel_size=1, padding=0),
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            ConvBlock(in_channels, 32, kernel_size=1, padding=0),
        )

        self.last_conv = nn.Sequential(
            ConvBlock(in_channels + 128, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        _, _, h, w = x.shape

        # Apply each pooling branch (fixed output sizes, then upsample back)
        branch1_out = self.branch1(x)
        branch1_out = F.interpolate(branch1_out, size=(h, w), mode='bilinear', align_corners=False)

        branch2_out = self.branch2(x)
        branch2_out = F.interpolate(branch2_out, size=(h, w), mode='bilinear', align_corners=False)

        branch3_out = self.branch3(x)
        branch3_out = F.interpolate(branch3_out, size=(h, w), mode='bilinear', align_corners=False)

        branch4_out = self.branch4(x)
        branch4_out = F.interpolate(branch4_out, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate along channel dimension
        concat = torch.cat([x, branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

        # Final convolution
        out = self.last_conv(concat)

        return out


class FeatureExtractor(nn.Module):
    """
    PSMNet-based Feature Extractor for cross-modal stereo matching.
    
    Extracts multi-scale features from input images (RGB or NIR).
    Outputs feature maps at 1/4 of the original resolution.
    
    Architecture:
        - Initial 2D conv layers (stride 2)
        - 4 ResNet basic blocks (stage 1-4)
        - Spatial Pyramid Pooling (SPP) module
        - Final conv to reduce channels
    """
    def __init__(self, in_channels=3, max_disparity=192):
        super(FeatureExtractor, self).__init__()
        self.max_disparity = max_disparity

        # Initial convolution layers (stride 2 to reduce resolution)
        self.conv_start = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # ResNet-like stages for feature extraction
        self.conv1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )

        # Dilated convolution blocks for larger receptive field
        self.conv3 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=8, dilation=8),
        )

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(128)

        # Final convolution to reduce to 32 channels
        self.conv_final = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, C, H, W)
               For RGB: C=3, for NIR: C=1
        
        Returns:
            features: Feature map (B, 32, H/4, W/4)
        """
        # Initial conv (H/2)
        out = self.conv_start(x)

        # Stage 1 (H/4)
        out = self.conv1(out)

        # Stage 2 (H/8)
        out = self.conv2(out)

        # Stage 3 - dilated convolutions (H/8)
        out = self.conv3(out)

        # Spatial Pyramid Pooling (H/8)
        out = self.spp(out)

        # Final conv to 32 channels
        features = self.conv_final(out)

        return features


class FeatureExtractorShared(nn.Module):
    """
    Shared-weight Feature Extractor for cross-modal matching.
    Uses the same FeatureExtractor for both RGB and NIR images,
    but the first conv layer handles different input channels
    via separate conv_start layers.
    """
    def __init__(self, rgb_channels=3, nir_channels=1, max_disparity=192):
        super(FeatureExtractorShared, self).__init__()

        # Separate initial layers for different input channels
        self.rgb_start = nn.Sequential(
            ConvBlock(rgb_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.nir_start = nn.Sequential(
            ConvBlock(nir_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # Shared deeper layers
        self.conv1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.conv3 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=8, dilation=8),
        )

        self.spp = SpatialPyramidPooling(128)

        self.conv_final = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_rgb(self, x):
        """Forward pass for RGB image (3 channels)"""
        out = self.rgb_start(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.spp(out)
        out = self.conv_final(out)
        return out

    def forward_nir(self, x):
        """Forward pass for NIR image (1 channel)"""
        out = self.nir_start(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.spp(out)
        out = self.conv_final(out)
        return out

    def forward(self, rgb, nir):
        """
        Forward pass for both modalities
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            nir: NIR image tensor (B, 1, H, W)
        
        Returns:
            f_rgb: RGB feature map (B, 32, H/4, W/4)
            f_nir: NIR feature map (B, 32, H/4, W/4)
        """
        f_rgb = self.forward_rgb(rgb)
        f_nir = self.forward_nir(nir)
        return f_rgb, f_nir


if __name__ == "__main__":
    # Simple test to verify the module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test FeatureExtractor (separate weights)
    model_rgb = FeatureExtractor(in_channels=3).to(device)
    model_nir = FeatureExtractor(in_channels=1).to(device)

    rgb_input = torch.randn(2, 3, 384, 640).to(device)
    nir_input = torch.randn(2, 1, 384, 640).to(device)

    with torch.no_grad():
        f_rgb = model_rgb(rgb_input)
        f_nir = model_nir(nir_input)

    print(f"RGB input:  {rgb_input.shape}")
    print(f"NIR input:  {nir_input.shape}")
    print(f"RGB feature: {f_rgb.shape}")
    print(f"NIR feature: {f_nir.shape}")
    print(f"Expected feature size: (2, 32, {384//4}, {640//4})")
    print(f"Feature extractor params: {sum(p.numel() for p in model_rgb.parameters()):,}")

    # Test shared weight version
    model_shared = FeatureExtractorShared().to(device)
    f_rgb_s, f_nir_s = model_shared(rgb_input, nir_input)
    print(f"\nShared model:")
    print(f"RGB feature: {f_rgb_s.shape}")
    print(f"NIR feature: {f_nir_s.shape}")
    print(f"Shared model params: {sum(p.numel() for p in model_shared.parameters()):,}")