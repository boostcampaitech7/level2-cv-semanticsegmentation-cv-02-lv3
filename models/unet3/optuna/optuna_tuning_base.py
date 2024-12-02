import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import *  # 위 코드의 UNet 3+ 모델 import
from loss import DiceLoss, TverskyLoss, FocalLoss, BCEWithIoUAndSSIM, FocalLossWithIoUAndSSIM
from dataset import *
import numpy as np
import albumentations as A
import time

# 데이터 로드 함수
def get_data_loaders(batch_size, image_root, label_root, split_file):
    # 변환 설정
    tf = A.Resize(512, 512)

    # 데이터셋 정의
    train_dataset = XRayDataset(
        image_root=image_root,
        label_root=label_root,
        is_train=True,
        transforms=tf,
        split_file=split_file,
    )

    valid_dataset = XRayDataset(
        image_root=image_root,
        label_root=label_root,
        is_train=False,
        transforms=tf,
        split_file=split_file,
    )

    # 데이터로더 정의
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    print("Train Loader Length:", len(train_loader))
    print("Valid Loader Length:", len(valid_loader))

    return train_loader, valid_loader

# 손실 함수 선택 함수
def get_loss_function(loss_type, alpha=0.7, gamma=2.0):
    if loss_type == "BCE":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "Dice":
        return DiceLoss()
    elif loss_type == "Focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "Hybrid_BCE":
        return BCEWithIoUAndSSIM()
    elif loss_type == "Hybrid_Focal":
        return FocalLossWithIoUAndSSIM()
    elif loss_type == "Tversky":
        return TverskyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def dice_score(y_true, y_pred):
    """
    Calculate Dice Score using dice_coef function.
    Args:
        y_true (torch.Tensor): 실제 레이블 (B, C, H, W)
        y_pred (torch.Tensor): 모델의 예측값 (B, C, H, W)
    Returns:
        float: 평균 Dice Score
    """
    y_true = y_true.flatten(2)  # (B, C, H*W)
    y_pred = torch.sigmoid(y_pred).flatten(2)  # (B, C, H*W)
    intersection = torch.sum(y_true * y_pred, dim=-1)
    eps = 0.0001
    dice = (2. * intersection + eps) / (torch.sum(y_true, dim=-1) + torch.sum(y_pred, dim=-1) + eps)
    return torch.mean(dice).item()


# Optuna Objective 함수
def objective(trial):
    # 1. 하이퍼파라미터 탐색
    print(f"Trial {trial.number} started.")
    start = time.time()
    loss_type = trial.suggest_categorical('loss_type', ["BCE", "Dice", "Focal", "Hybrid_BCE", "Hybrid_Focal", "Tversky"])
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    batch_size = trial.suggest_categorical('batch_size', [2])
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    # dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    # 데이터 로드
    image_root = '/data/ephemeral/home/data/train/DCM'
    label_root = '/data/ephemeral/home/data/train/outputs_json'
    split_file = f'/data/ephemeral/home/Jihwan/split/fold_0.json'
    train_loader, valid_loader = get_data_loaders(batch_size, image_root, label_root, split_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 초기화
    model = UNet_3Plus(
        in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True
    ).to(device)
    # 손실 함수 및 Optimizer 설정
    criterion = get_loss_function(loss_type, alpha=alpha, gamma=gamma)
    optimizer = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop
    }[optimizer_type](model.parameters(), lr=lr, weight_decay=weight_decay)
    # 학습 루프
    epochs = 5
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            outputs = torch.nn.functional.interpolate(outputs, size=y.shape[-2:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    # 검증 단계에서 Dice Score 계산
    model.eval()
    total_dice_score = 0.0
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            outputs = torch.nn.functional.interpolate(outputs, size=y.shape[-2:], mode='bilinear', align_corners=True)
            total_dice_score += dice_score(y, outputs)
    avg_dice_score = total_dice_score / len(valid_loader)
    end = time.time()
    elapsed_time = start - end
    print(f"Trial {trial.number} completed. Dice Score: {avg_dice_score:.4f}. Execution time: {elapsed_time:.2f} seconds")

    return avg_dice_score  # Optuna가 최대화하도록 설정


# Optuna 실행
if __name__ == "__main__":
    
    import optuna.visualization as vis
    # Optuna Study 생성
    study = optuna.create_study(
        direction="maximize",  # Dice Score를 최대화
        storage="sqlite:///optuna_study.db",
        study_name="example_study",
        load_if_exists=True
    )
    print("Starting Optuna study...")
    # 최적화 실행
    study.optimize(objective, n_trials=30)
    print("Study completed.")
    print("Best parameters:", study.best_params)
    print("Best validation Dice Score:", study.best_value)
    # 파라미터 중요도 확인 그래프
    print("Plotting parameter importances...")
    param_importances_fig = vis.plot_param_importances(study)
    param_importances_fig.show()
    # 최적화 과정 시각화
    print("Plotting optimization history...")
    optimization_history_fig = vis.plot_optimization_history(study)
    optimization_history_fig.show()