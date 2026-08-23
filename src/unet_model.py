import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two consecutive Convolution -> BatchNorm -> ReLU layers.
    Uses padding=0 (valid convolutions) as in original Ronneberger et al. (2015).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def crop_tensor(enc_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Crops encoder feature map to match spatial dimensions of upsampled decoder tensor for skip connection concatenation.
    """
    _, _, H_enc, W_enc = enc_tensor.shape
    _, _, H_tgt, W_tgt = target_tensor.shape
    delta_H = (H_enc - H_tgt) // 2
    delta_W = (W_enc - W_tgt) // 2
    return enc_tensor[:, :, delta_H : delta_H + H_tgt, delta_W : delta_W + W_tgt]

class UNet(nn.Module):
    """
    Original U-Net Architecture (Ronneberger et al., 2015) with valid convolutions (padding=0).
    
    Input shape:  (B, in_channels, 572, 572)
    Output shape: (B, out_channels, 388, 388)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        
        # Contracting Path (Encoder)
        self.enc1 = DoubleConv(in_channels, base_filters)           # 1 -> 64 (572 -> 568)
        self.pool1 = nn.MaxPool2d(2, 2)                            # 568 -> 284
        
        self.enc2 = DoubleConv(base_filters, base_filters * 2)       # 64 -> 128 (284 -> 280)
        self.pool2 = nn.MaxPool2d(2, 2)                            # 280 -> 140
        
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)   # 128 -> 256 (140 -> 136)
        self.pool3 = nn.MaxPool2d(2, 2)                            # 136 -> 68
        
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)   # 256 -> 512 (68 -> 64)
        self.pool4 = nn.MaxPool2d(2, 2)                            # 64 -> 32
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16) # 512 -> 1024 (32 -> 28)
        self.dropout = nn.Dropout(0.5)
        
        # Expansive Path (Decoder)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2) # 28 -> 56
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8) # (512+512) -> 512 (56 -> 52)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)  # 52 -> 104
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)  # (256+256) -> 256 (104 -> 100)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)  # 100 -> 200
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)  # (128+128) -> 128 (200 -> 196)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)      # 196 -> 392
        self.dec1 = DoubleConv(base_filters * 2, base_filters)      # (64+64) -> 64 (392 -> 388)
        
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
        
        # Decoder with Crop & Concatenate Skip Connections
        u4 = self.up4(b)
        e4_cropped = crop_tensor(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4_cropped], dim=1))
        
        u3 = self.up3(d4)
        e3_cropped = crop_tensor(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3_cropped], dim=1))
        
        u2 = self.up2(d3)
        e2_cropped = crop_tensor(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_cropped], dim=1))
        
        u1 = self.up1(d2)
        e1_cropped = crop_tensor(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1_cropped], dim=1))
        
        logits = self.out_conv(d1)
        output = self.sigmoid(logits)
        return output

