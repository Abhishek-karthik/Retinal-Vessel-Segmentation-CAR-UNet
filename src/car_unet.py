"""
Guo, C. et al. "Channel Attention Residual U-Net for Retinal Vessel Segmentation." IEEE, 2021. https://arxiv.org/abs/2004.03702
Official CAR-UNet Reference: https://github.com/clguo/CAR-UNet

PyTorch Implementation of CAR-UNet (Channel Attention Residual U-Net):
1. MECA (Modified Efficient Channel Attention): Channel attention mechanism using 1D convolution across channels.
2. CADRB (Channel Attention Double Residual Block): Residual convolutional block enhanced with MECA channel attention.
3. Bridge Attention: Skip connections filtered by MECA modules before concatenation with decoder features.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MECA(nn.Module):
    """
    Modified Efficient Channel Attention (MECA) Module.
    Uses 1D convolution with adaptive kernel size across channels to capture
    local cross-channel interactions without dimensionality reduction.
    Equation: k = |(log2(C) + b) / gamma|_odd
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1, k_size: int = None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(max(channels, 2)) + b) / gamma))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(3, k_size) # ensure minimum odd kernel size >= 3
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x) # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

def crop_tensor(enc_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Crops encoder feature map to match spatial dimensions of upsampled decoder tensor for skip connection concatenation.
    """
    _, _, H_enc, W_enc = enc_tensor.shape
    _, _, H_tgt, W_tgt = target_tensor.shape
    delta_H = (H_enc - H_tgt) // 2
    delta_W = (W_enc - W_tgt) // 2
    return enc_tensor[:, :, delta_H : delta_H + H_tgt, delta_W : delta_W + W_tgt]

class CADRB(nn.Module):
    """
    Channel Attention Double Residual Block (CADRB / CARB).
    Combines two convolutions, BatchNorm, MECA channel attention, and a residual skip connection.
    Supports valid convolutions (padding=0) matching Ronneberger U-Net geometry.
    """
    def __init__(self, in_channels: int, out_channels: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.meca = MECA(out_channels)
        
        # Shortcut / Residual path
        self.need_proj = (in_channels != out_channels)
        if self.need_proj:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.meca(out) # Apply Channel Attention
        
        # Residual branch
        if self.need_proj:
            res = self.shortcut_bn(self.shortcut_conv(x))
        else:
            res = x
            
        # If valid convolutions (padding=0), crop residual to match spatial dimension of main branch
        if self.padding == 0:
            res = crop_tensor(res, out)
            
        out = self.relu(out + res)
        return out

class CARUNet(nn.Module):
    """
    Channel Attention Residual U-Net (CAR-UNet) for Retinal Vessel Segmentation.
    
    Key Paper Additions over Plain U-Net:
    1. Channel Attention Double Residual Blocks (CADRB) with MECA in all encoder/decoder stages.
    2. Modified Efficient Channel Attention (MECA) on skip connections before concatenation.
    3. Residual learning prevents vanishing gradients and enhances vessel edge feature propagation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        
        # Contracting Path (Encoder with CADRB)
        self.enc1 = CADRB(in_channels, base_filters, padding=0)               # 1 -> 64 (572 -> 568)
        self.pool1 = nn.MaxPool2d(2, 2)                                      # 568 -> 284
        self.skip_att1 = MECA(base_filters)                                  # Skip Connection Attention 1
        
        self.enc2 = CADRB(base_filters, base_filters * 2, padding=0)         # 64 -> 128 (284 -> 280)
        self.pool2 = nn.MaxPool2d(2, 2)                                      # 280 -> 140
        self.skip_att2 = MECA(base_filters * 2)                              # Skip Connection Attention 2
        
        self.enc3 = CADRB(base_filters * 2, base_filters * 4, padding=0)     # 128 -> 256 (140 -> 136)
        self.pool3 = nn.MaxPool2d(2, 2)                                      # 136 -> 68
        self.skip_att3 = MECA(base_filters * 4)                              # Skip Connection Attention 3
        
        self.enc4 = CADRB(base_filters * 4, base_filters * 8, padding=0)     # 256 -> 512 (68 -> 64)
        self.pool4 = nn.MaxPool2d(2, 2)                                      # 64 -> 32
        self.skip_att4 = MECA(base_filters * 8)                              # Skip Connection Attention 4
        
        # Bottleneck
        self.bottleneck = CADRB(base_filters * 8, base_filters * 16, padding=0) # 512 -> 1024 (32 -> 28)
        self.dropout = nn.Dropout(0.5)
        
        # Expansive Path (Decoder with CADRB)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2) # 28 -> 56
        self.dec4 = CADRB(base_filters * 16, base_filters * 8, padding=0) # (512+512) -> 512 (56 -> 52)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)  # 52 -> 104
        self.dec3 = CADRB(base_filters * 8, base_filters * 4, padding=0)  # (256+256) -> 256 (104 -> 100)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)  # 100 -> 200
        self.dec2 = CADRB(base_filters * 4, base_filters * 2, padding=0)  # (128+128) -> 128 (200 -> 196)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)      # 196 -> 392
        self.dec1 = CADRB(base_filters * 2, base_filters, padding=0)      # (64+64) -> 64 (392 -> 388)
        
        # Final Output Layer
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.dropout(self.bottleneck(self.pool4(e4)))
        
        # Decoder with MECA-Attentive Skip Connections
        u4 = self.up4(b)
        e4_att = self.skip_att4(e4)
        e4_cropped = crop_tensor(e4_att, u4)
        d4 = self.dec4(torch.cat([u4, e4_cropped], dim=1))
        
        u3 = self.up3(d4)
        e3_att = self.skip_att3(e3)
        e3_cropped = crop_tensor(e3_att, u3)
        d3 = self.dec3(torch.cat([u3, e3_cropped], dim=1))
        
        u2 = self.up2(d3)
        e2_att = self.skip_att2(e2)
        e2_cropped = crop_tensor(e2_att, u2)
        d2 = self.dec2(torch.cat([u2, e2_cropped], dim=1))
        
        u1 = self.up1(d2)
        e1_att = self.skip_att1(e1)
        e1_cropped = crop_tensor(e1_att, u1)
        d1 = self.dec1(torch.cat([u1, e1_cropped], dim=1))
        
        logits = self.out_conv(d1)
        output = self.sigmoid(logits)
        return output

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing CAR-UNet Architecture...")
    model = CARUNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    
    # Test valid convolution patch: 284x284 -> 100x100
    dummy_input = torch.randn(2, 1, 284, 284, device=device)
    dummy_output = model(dummy_input)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1, 100, 100), f"Expected (2, 1, 100, 100), got {dummy_output.shape}"
    assert (dummy_output >= 0.0).all() and (dummy_output <= 1.0).all(), "Output not in [0, 1] range"
    print(">>> CAR-UNet Sanity Check Passed Perfectly!")
