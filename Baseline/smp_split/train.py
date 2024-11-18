# python native
import os
import json
import random
import datetime
from functools import partial
import segmentation_models_pytorch as smp
from dotenv import load_dotenv
# pip install git+https://github.com/qubvel/segmentation_models.pytorch

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt
import argparse
from dataset import XRayDataset
from trainer import *
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='valid Batch size')
    parser.add_argument('--val_every', type=int, default=10, help='val_every')
    parser.add_argument('--image_root', type=str,default='/data/ephemeral/home/data/train/DCM',help='image root')
    parser.add_argument('--label_root',type=str,default='/data/ephemeral/home/data/train/outputs_json',help='label root')
    parser.add_argument('--saved_dir', type=str, default='checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='resnet101', help='Name of the segmentation model')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='encoder weights')
    parser.add_argument('--seg_model', type=str, default='UnetPlusPlus', help='Segmentation model name')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help='Resize dimensions: height width')
    parser.add_argument('--json_dir', type=str, default='../Data/train_valid_split/splits', help='train_valid_split_dir')
    args = parser.parse_args()

    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)

    # wandb 초기화
    wandb.init(entity="luckyvicky",project="segmentation", name=f"{args.seg_model}_{args.model_name}_{args.resize}_batch{args.batch_size}",config={
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "model_name": args.model_name
    })
    
if not os.path.exists(args.saved_dir):                                                           
    os.makedirs(args.saved_dir)

RANDOM_SEED = 21

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# 특정 fold 파일 불러오기
fold_idx = 0  # 원하는 fold index를 지정
# json 파일 경로 지정
split_file = args.json_dir+f'/fold_{fold_idx}.json'


# dataset
#tf = A.Resize(512, 512)
resize_height, resize_width = args.resize
tf = A.Resize(resize_height, resize_width)
train_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=True, transforms=tf)
valid_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=False, transforms=tf)

# dataloader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=args.valid_batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False
)


# model
seg_model_name = getattr(smp, args.seg_model, None)

if seg_model_name:
    model = seg_model_name(
        encoder_name=args.model_name,         # Encoder 이름 (e.g., resnet101, efficientnet-b0)
        encoder_weights=args.encoder_weights, # Pretrained weights (e.g., imagenet)
        in_channels=3,                        # 입력 채널 (e.g., RGB)
        classes=len(CLASSES)                  # 출력 클래스 수
    )
else:
    raise ValueError(f"Segmentation model '{args.seg_model}' is not available in smp.")

# Loss function을 정의합니다.
criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

# 시드를 설정합니다.
set_seed()

train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.val_every, args.saved_dir, args.model_name, args.seg_model, args.resize, args.batch_size)

wandb.finish()


# python train.py