import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

# BCE Loss
def BCE_loss(pred, label):
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # Sigmoid 포함
    bce_out = bce_loss(pred, label)
    return bce_out


def focal_loss(y_pred, label, gamma=2.0, alpha=4.0, epsilon=1e-9):
    label = label.float()
    y_pred = torch.sigmoid(y_pred)  # Focal Loss에도 Sigmoid 적용
    ce = -label * torch.log(y_pred + epsilon)  # Cross-Entropy
    weight = (1 - y_pred) ** gamma  # 가중치 계산
    fl = alpha * weight * ce
    return fl.mean()  # 샘플별 손실 평균


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pt = target * pred + (1 - target) * (1 - pred)  # pt = p if target = 1 else (1-p)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt + 1e-6)  # Prevent log(0)
        return loss.mean()


def combined_bce_focal_loss(pred, label, class_weights, focal_weight=0.5, gamma=2.0, alpha=0.7, epsilon=1e-9):
    """
    BCE Loss (클래스 가중치 포함)와 Focal Loss를 결합한 손실 함수.

    Args:
        pred (torch.Tensor): 모델의 예측값 (logits, shape: (N, C, H, W)).
        label (torch.Tensor): 타겟값 (shape: (N, C, H, W)).
        class_weights (torch.Tensor): 각 클래스의 가중치 텐서 (shape: (1, C, 1, 1)).
        focal_weight (float): Focal Loss의 가중치.
        gamma (float): Focal Loss의 감쇠 파라미터.
        alpha (float): Focal Loss의 균형 파라미터.
        epsilon (float): 수치 안정성을 위한 작은 값.

    Returns:
        torch.Tensor: 결합 손실 값 (scalar).
    """
    # BCE Loss 계산 (클래스 가중치 적용)
    bce_loss = F.binary_cross_entropy_with_logits(
        pred, label, weight=class_weights, reduction='none'
    )
    bce_loss = bce_loss.mean()

    # Focal Loss 계산
    label = label.float()
    y_pred = torch.sigmoid(pred)  # Focal Loss에도 Sigmoid 적용
    ce = -label * torch.log(y_pred + epsilon)  # Cross-Entropy
    weight = (1 - y_pred) ** gamma  # Focal Loss 가중치 계산
    focal_loss = alpha * weight * ce
    focal_loss = focal_loss.mean()

    # BCE Loss와 Focal Loss 결합
    combined_loss = (1 - focal_weight) * bce_loss + focal_weight * focal_loss

    return combined_loss


# IoU Loss 계산 함수
def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(b):
        Iand1 = torch.sum(target[i] * pred[i])
        Ior1 = torch.sum(target[i]) + torch.sum(pred[i]) - Iand1
        IoU1 = Iand1 / Ior1
        IoU += (1 - IoU1)  # IoU Loss
    return IoU / b

# IoU Loss 클래스
class IOU(nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)

# IoU Loss 함수
def IOU_loss(pred, label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    #print("iou_loss:", iou_out.item())  # detach().item() 대신 item() 사용
    return iou_out

# Gaussian Kernel 생성 함수
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# 윈도우 생성 함수
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# SSIM 계산 함수
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

# MSSSIM 계산 함수
def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

# MSSSIM Loss
def msssim_loss(pred, label):
    msssim_val = msssim(pred, label, size_average=True)
    msssim_loss = torch.clamp(1 - msssim_val, min=0, max=1)
    #print("msssim_loss:", msssim_loss.item())
    return msssim_loss

##################################################################################
#combined_bce_dice위해 이걸 사용할 것
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def combined_bce_dice(pred, target, class_weights, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target, weight=class_weights)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss
###################################################################################

# Dice Loss 계산 함수
#def dice_loss(pred, label, smooth=1.0):
#    """
#    Dice Loss
#    Args:
#        pred (torch.Tensor): 모델의 예측값 (B, C, H, W) 또는 (B, 1, H, W)
#        label (torch.Tensor): 실제 레이블 (B, C, H, W) 또는 (B, 1, H, W)
#        smooth (float): smoothing factor to avoid division by zero
#    Returns:
#        torch.Tensor: Dice Loss
#    """
#    pred = torch.sigmoid(pred)  # 예측값에 sigmoid 적용
#    intersection = torch.sum(pred * label, dim=(2, 3))
#    union = torch.sum(pred, dim=(2, 3)) + torch.sum(label, dim=(2, 3))
#    dice_score = (2.0 * intersection + smooth) / (union + smooth)
#    dice_loss = 1.0 - dice_score  # Dice Coefficient를 1에서 뺌 (Loss)
#    return torch.mean(dice_loss)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=0.001): 
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = torch.sum(pred * target, dim=(2, 3))
        fp = torch.sum(pred * (1 - target), dim=(2, 3))
        fn = torch.sum((1 - pred) * target, dim=(2, 3))
        tversky_score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_score.mean()



######결합#############3

# Dynamic weights 계산
def dynamic_weights(losses, epsilon=1e-6):
    losses = np.array(losses)  # numpy로 처리
    weights = 1 / (losses + epsilon)
    return weights / weights.sum()

# BCE + iou + mssim
def combined_loss_with_dynamic_weights(pred, label):
    # Label 값을 [0, 1]로 제한
    label = torch.clamp(label, 0, 1)

    # 개별 손실 계산
    bce = BCE_loss(pred, label)
    iou = IOU_loss(torch.sigmoid(pred), label)  # IoU에 sigmoid 적용
    msssim = msssim_loss(torch.sigmoid(pred), label)  # MSSSIM에 sigmoid 적용

    # Dynamic Weight 계산
    losses = [bce.item(), msssim.item(), iou.item()]
    weights = dynamic_weights(losses)

    #print(f"pred shape: {pred.shape}, label shape: {label.shape}")
    #print(f"pred range: {pred.min().item()} to {pred.max().item()}, label range: {label.min().item()} to {label.max().item()}")
    #print(f"Weights: {weights.tolist()}")

    # Combined Loss
    total_loss = weights[0] * bce + weights[1] * msssim + weights[2] * iou
    return total_loss

# BCE + IoU + MSSSIM Loss
class BCEWithIoUAndSSIM(nn.Module):
    def __init__(self):
        super(BCEWithIoUAndSSIM, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        iou = IOU_loss(torch.sigmoid(pred), target)
        msssim = msssim_loss(torch.sigmoid(pred), target)

        # 동적 가중치 계산
        losses = [bce.item(), iou.item(), msssim.item()]
        weights = dynamic_weights(losses)

        # 최종 손실 계산
        total_loss = weights[0] * bce + weights[1] * iou + weights[2] * msssim
        return total_loss

# Focal + IoU + MSSSIM Loss
class FocalLossWithIoUAndSSIM(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLossWithIoUAndSSIM, self).__init__()
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, pred, target):
        focal = self.focal(pred, target)
        iou = IOU_loss(torch.sigmoid(pred), target)
        msssim = msssim_loss(torch.sigmoid(pred), target)

        # 동적 가중치 계산
        losses = [focal.item(), iou.item(), msssim.item()]
        weights = dynamic_weights(losses)

        # 최종 손실 계산
        total_loss = weights[0] * focal + weights[1] * iou + weights[2] * msssim
        return total_loss

