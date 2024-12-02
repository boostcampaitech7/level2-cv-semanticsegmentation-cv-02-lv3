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

# Optuna Objective 함수
def objective(trial):
    # 1. 하이퍼파라미터 탐색
    print(f"Trial {trial.number} started.")

    w1 = trial.suggest_float('w1', 0.0, 1.0)
    w2 = trial.suggest_float('w2', 0.0, 1.0)
    w3 = trial.suggest_float('w3', 0.0, 1.0)
    w4 = trial.suggest_float('w4', 0.0, 1.0)
    w5 = trial.suggest_float('w5', 0.0, 1.0)
    loss_type = trial.suggest_categorical('loss_type', ["BCE", "Dice", "Focal", "Hybrid_BCE", "Hybrid_Focal", "Tversky"])
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    batch_size = trial.suggest_categorical('batch_size', [2])
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4)

    # Focal Loss 관련 하이퍼파라미터
    alpha = trial.suggest_float('alpha', 0.1, 1.0)  # alpha 탐색 범위
    gamma = trial.suggest_float('gamma', 1.0, 3.0)  # gamma 탐색 범위

    # 가중치 정규화
    total = sum([w1, w2, w3, w4, w5]) or 1.0
    w1, w2, w3, w4, w5 = [w / total for w in [w1, w2, w3, w4, w5]]

    # 데이터 로드
    image_root = '/data/ephemeral/home/data/train/DCM'
    label_root = '/data/ephemeral/home/data/train/outputs_json'
    split_file = f'/data/ephemeral/home/Jihwan/split/fold_0.json'  # Fold 0 사용 예시
    train_loader, valid_loader = get_data_loaders(batch_size, image_root, label_root, split_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 모델 초기화
    model = UNet_3Plus_DeepSup(
        in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True
    ).to(device)

    # 4. 손실 함수 및 Optimizer 설정
    criterion = get_loss_function(loss_type, alpha=alpha, gamma=gamma)

    optimizer = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop
    }[optimizer_type](model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. 학습 루프
    epochs = 4  # 간단한 테스트용, 필요 시 증가
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started.")
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # 모델 출력
            d1, d2, d3, d4, d5 = model(X)
            outputs = sum([w * d for w, d in zip([w1, w2, w3, w4, w5], [d1, d2, d3, d4, d5])])
            # 출력 크기를 라벨 크기에 맞춤
            outputs = torch.nn.functional.interpolate(outputs, size=y.shape[-2:], mode='bilinear', align_corners=True)
            # 손실 계산 및 역전파
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        

    # 6. 검증 손실 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            # 모델 출력
            d1, d2, d3, d4, d5 = model(X)
            outputs = sum([w * d for w, d in zip([w1, w2, w3, w4, w5], [d1, d2, d3, d4, d5])])
            # 출력 크기를 라벨 크기에 맞춤
            outputs = torch.nn.functional.interpolate(outputs, size=y.shape[-2:], mode='bilinear', align_corners=True)
            # 손실 계산
            val_loss += criterion(outputs, y).item()

    return val_loss / len(valid_loader)

    


# Optuna 실행
if __name__ == "__main__":

    study = optuna.create_study(
    direction="minimize",
    storage="sqlite:///optuna_study.db",  # SQLite 데이터베이스 파일
    study_name="example_study",          # Study 이름
    load_if_exists=True                  # 동일한 이름의 Study가 있으면 로드
    )

    print("Starting Optuna study...")
    study.optimize(objective, n_trials=10)
    print("Study completed.")

    # 최적의 하이퍼파라미터 출력
    print("Best parameters:", study.best_params)
    print("Best validation loss:", study.best_value)
