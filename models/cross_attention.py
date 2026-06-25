import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer between two modalities.
    
    Formula: F_aligned = softmax(Q·K^T / sqrt(d)) · V
    
    Layer 1: Q=RGB, K=NIR, V=NIR -> NIR features attend to RGB query
    Layer 2: Q=NIR, K=RGB, V=RGB -> RGB features attend to NIR query
    """
    def __init__(self, channels):
        super(CrossAttentionLayer, self).__init__()
        self.channels = channels
        self.scale = channels ** -0.5  # 1 / sqrt(d)

        # Linear projections for Q, K, V
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * self.scale)

    def forward(self, x_query, x_key):
        """
        Args:
            x_query: Query features (B, C, H, W) - from modality 1
            x_key:   Key/Value features (B, C, H, W) - from modality 2
        
        Returns:
            out: Aligned features (B, C, H, W)
        """
        B, C, H, W = x_query.shape

        Q = self.q_proj(x_query)
        K = self.k_proj(x_key)
        V = self.v_proj(x_key)

        Q = Q.view(B, C, -1)
        K = K.view(B, C, -1)
        V = V.view(B, C, -1)

        K_T = K.transpose(-2, -1)

        attn = torch.bmm(Q, K_T)
        attn = attn * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        out = out + x_key

        return out


class SpatialCrossAttentionLayer(nn.Module):
    """
    Spatial cross-attention (pixel-to-pixel) between two modalities.
    """
    def __init__(self, channels):
        super(SpatialCrossAttentionLayer, self).__init__()
        self.channels = channels
        self.scale = channels ** -0.5

        self.q_proj = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x_query, x_key):
        B, C, H, W = x_query.shape
        N = H * W

        Q = self.q_proj(x_query).view(B, -1, N)
        K = self.k_proj(x_key).view(B, -1, N)
        V = self.v_proj(x_key).view(B, C, N)

        attn = torch.bmm(Q.transpose(-2, -1), K) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.transpose(-2, -1))
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        out = out + x_key
        return out


class CrossAttentionModule(nn.Module):
    """
    Cross-modal Feature Matching (CFM) attention module from the paper.
    
    Applies two cross-attention layers bidirectionally:
    - Layer 1: RGB -> NIR (RGB as query, NIR as key/value)
    - Layer 2: NIR -> RGB (NIR as query, RGB as key/value)
    """
    def __init__(self, channels=32, use_spatial=False):
        super(CrossAttentionModule, self).__init__()
        self.use_spatial = use_spatial

        AttentionLayer = SpatialCrossAttentionLayer if use_spatial else CrossAttentionLayer

        self.attn_rgb2nir = AttentionLayer(channels)
        self.attn_nir2rgb = AttentionLayer(channels)

        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_nir = nn.LayerNorm(channels)

    def forward(self, f_rgb, f_nir):
        B, C, H, W = f_rgb.shape

        f_nir_aligned = self.attn_rgb2nir(f_rgb, f_nir)
        f_rgb_aligned = self.attn_nir2rgb(f_nir, f_rgb)

        f_rgb_aligned = f_rgb_aligned.permute(0, 2, 3, 1).contiguous().view(-1, C)
        f_rgb_aligned = self.norm_rgb(f_rgb_aligned).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        f_nir_aligned = f_nir_aligned.permute(0, 2, 3, 1).contiguous().view(-1, C)
        f_nir_aligned = self.norm_nir(f_nir_aligned).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return f_rgb_aligned, f_nir_aligned


class CrossAttentionModuleSimple(nn.Module):
    """Simplified cross-attention using 1x1 conv alignment."""
    def __init__(self, channels=32):
        super(CrossAttentionModuleSimple, self).__init__()
        self.proj_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.proj_nir = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb, f_nir):
        f_rgb_aligned = self.proj_rgb(f_rgb)
        f_nir_aligned = self.proj_nir(f_nir)
        return f_rgb_aligned, f_nir_aligned


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, C, H, W = 2, 32, 96, 160
    f_rgb = torch.randn(B, C, H, W).to(device)
    f_nir = torch.randn(B, C, H, W).to(device)

    # Test channel attention version
    model = CrossAttentionModule(channels=C).to(device)
    f_rgb_out, f_nir_out = model(f_rgb, f_nir)
    print(f"Input RGB:     {f_rgb.shape}")
    print(f"Input NIR:     {f_nir.shape}")
    print(f"Output RGB:    {f_rgb_out.shape}")
    print(f"Output NIR:    {f_nir_out.shape}")
    print(f"Params:        {sum(p.numel() for p in model.parameters()):,}")
    assert f_rgb_out.shape == f_rgb.shape
    assert f_nir_out.shape == f_nir.shape
    print("All tests passed!")

    # Test simple version
    model_simple = CrossAttentionModuleSimple(channels=C).to(device)
    f_rgb_s, f_nir_s = model_simple(f_rgb, f_nir)
    print(f"\nSimple version:")
    print(f"Output RGB:    {f_rgb_s.shape}")
    print(f"Output NIR:    {f_nir_s.shape}")
    print(f"Params:        {sum(p.numel() for p in model_simple.parameters()):,}")

    # Test spatial attention version
    model_spatial = CrossAttentionModule(channels=C, use_spatial=True).to(device)
    f_rgb_sp, f_nir_sp = model_spatial(f_rgb, f_nir)
    print(f"\nSpatial attention version:")
    print(f"Output RGB:    {f_rgb_sp.shape}")
    print(f"Output NIR:    {f_nir_sp.shape}")
    print(f"Params:        {sum(p.numel() for p in model_spatial.parameters()):,}")