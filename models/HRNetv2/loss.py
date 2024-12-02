import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, weight_bce=0.7, weight_focal=0.3):
        """
        Combines BCEWithLogitsLoss and Focal Loss.

        :param alpha: Alpha value for Focal Loss.
        :param gamma: Gamma value for Focal Loss.
        :param weight_bce: Weight of BCE loss in the hybrid loss.
        :param weight_focal: Weight of Focal loss in the hybrid loss.
        """
        super(HybridLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_bce = weight_bce
        self.weight_focal = weight_focal

    def forward(self, logits, targets):
        # Compute BCE Loss
        bce_loss = self.bce_loss(logits, targets)

        # Compute Focal Loss
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * targets * torch.log(probs + 1e-9) - \
                     (1 - self.alpha) * pt ** self.gamma * (1 - targets) * torch.log(1 - probs + 1e-9)
        focal_loss = focal_loss.mean()

        # Combine the losses
        hybrid_loss = self.weight_bce * bce_loss + self.weight_focal * focal_loss
        return hybrid_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        """
        Combines BCEWithLogitsLoss and Dice Loss.

        :param weight_bce: Weight of BCE loss in the combined loss.
        :param weight_dice: Weight of Dice loss in the combined loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def dice_loss(self, logits, targets, smooth=1e-6):
        """
        Computes Dice Loss.

        :param logits: Predicted logits from the model.
        :param targets: Ground truth masks.
        :param smooth: Smoothing factor to prevent division by zero.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(1)
        dice_score = (2. * intersection + smooth) / (probs_flat.sum(1) + targets_flat.sum(1) + smooth)
        return 1 - dice_score.mean()

    def forward(self, logits, targets):
        """
        Forward pass to compute combined loss.

        :param logits: Predicted logits from the model.
        :param targets: Ground truth masks.
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.weight_bce * bce + self.weight_dice * dice
    

class BCEFocalDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.4, weight_focal=0.2, gamma=2.0):
        super(BCEFocalDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.gamma = gamma

    def dice_loss(self, logits, targets, smooth=1e-6):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(1)
        dice_score = (2. * intersection + smooth) / (probs_flat.sum(1) + targets_flat.sum(1) + smooth)
        return 1 - dice_score.mean()

    def focal_loss(self, logits, targets, alpha=0.25, smooth=1e-6):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = -alpha * (1 - pt).pow(self.gamma) * torch.log(pt + smooth)
        return focal.mean()

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.weight_bce * bce + self.weight_dice * dice + self.weight_focal * focal