import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary image segmentation.
    Handles extreme class imbalance (vessels represent ~10-15% of pixels).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice_score

class CombinedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) + Dice Loss.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    # Minimum over a 3x3 cross (min-pooling = -maxpool(-x)), as in Shit et al. (2021)
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_skeleton(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton of a probability map (Shit et al., CVPR 2021)."""
    skel = F.relu(img - _soft_dilate(_soft_erode(img)))
    for _ in range(iterations):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_dilate(_soft_erode(img)))
        skel = skel + F.relu(delta - skel * delta)
    return skel

class SoftClDiceLoss(nn.Module):
    """
    clDice loss (Shit et al., "clDice - a Novel Topology-Preserving Loss Function for Tubular
    Structure Segmentation", CVPR 2021). Compares vessel SKELETONS instead of pixels, so
    breaking a thin vessel or missing a capillary costs as much as missing a thick vessel.
        Tprec = |S(pred) * GT| / |S(pred)|,  Tsens = |S(GT) * pred| / |S(GT)|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens),  loss = 1 - clDice
    """
    def __init__(self, iterations: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeleton(pred, self.iterations)
        skel_true = soft_skeleton(target, self.iterations)
        tprec = ((skel_pred * target).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((skel_true * pred).sum() + self.smooth) / (skel_true.sum() + self.smooth)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

class BCEDiceClDiceLoss(nn.Module):
    """
    Improvement 2: 0.5 * BCE + 0.5 * [(1 - alpha) * Dice + alpha * clDice].
    With alpha = 0 this is exactly CombinedBCEDiceLoss; alpha = 0.5 follows Shit et al. (2021).
    """
    def __init__(self, alpha: float = 0.5, iterations: int = 10, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)
        self.cldice = SoftClDiceLoss(iterations=iterations, smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        overlap = (1 - self.alpha) * self.dice(pred, target) + self.alpha * self.cldice(pred, target)
        return 0.5 * self.bce(pred, target) + 0.5 * overlap
