# python native
import os
import json
import random
import datetime
from functools import partial
from dotenv import load_dotenv
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
from model import *

import pdb

from loss import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size') 
    parser.add_argument('--valid_batch_size', type=int, default=1, help='valid Batch size') 
    parser.add_argument('--val_every', type=int, default=1, help='val_every')
    parser.add_argument('--image_root', type=str,default='/data/ephemeral/home/data/train/DCM',help='image root')
    parser.add_argument('--label_root',type=str,default='/data/ephemeral/home/data/train/outputs_json',help='label root')
    parser.add_argument('--saved_dir', type=str, default='checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='unet+++', help='Name of the segmentation model')
    parser.add_argument('--json_dir', type=str, default='../splits', help='train_valid_split_dir')
    parser.add_argument('--fold', type=int, default=0, help='split_k_fold_0')
    parser.add_argument('--model_class', type=str, default='UNet_3Plus', 
                        choices=['UNet_3Plus', 'UNet_3Plus_DeepSup', 'UNet_3Plus_DeepSup_CGM'], 
                        help='Model class to use')
    parser.add_argument('--loss_function', type=str, default='bce', 
                        choices=['bce', 'bce+iou+ssim', 'focal+iou+ssim', 'tversky', 'bce+focal', 'bce+dice'], 
                        help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'rmsprop'], 
                        help='Optimizer to use: adam or rmsprop')
    parser.add_argument('--class_weights', type=int, default=3, 
                        choices=[1, 2, 3], 
                        help='Class weight scheme to use (1, 2, or 3)')
    parser.add_argument('--image_size', type=int, default=512, 
                        choices=[448, 512, 616, 768, 1024], 
                        help='Image size (default: 512)')
    parser.add_argument(
        '--augmentations',
        type=str,
        nargs="*",
        default=[],
        choices=['grid', 'contrast', 'clahe'],
        help='Augmentations to apply during training (e.g., grid, contrast, clahe)'
    )
    
    args = parser.parse_args()

    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)

    # wandb 초기화
    wandb.init(entity="luckyvicky",project="segmentation", name=args.model_name,config={
       "epochs": args.epochs,
       "learning_rate": args.lr,
       "batch_size": args.batch_size,
       "valid_batch_size": args.valid_batch_size,
       "model_name": args.model_name
    })


if not os.path.exists(args.saved_dir):                                                           
    os.makedirs(args.saved_dir)

RANDOM_SEED = 42  #21

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

# json 파일 경로 지정(수정)
split_file = args.json_dir+f'/fold_{args.fold}.json'

# dataset
augmentations = []                                               # augmentation 조합 넣을 리스트
augmentations.append(A.Resize(args.image_size, args.image_size)) # Resize는 항상 추가
if 'grid' in args.augmentations:
    augmentations.append(A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5))
if 'contraast' in args.augmentations:
    augmentations.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, 0.5), p=0.5))
if 'clahe' in args.augmentations:
    augmentations.append(A.CLAHE(clip_limit=(1, 2), p=0.7))

# train transform (augmentation 동적으로 조합)
tf_train = A.Compose(augmentations)

# valid transform (기본은 Resize만 적용)
tf_valid = A.Resize(args.image_size, args.image_size)


train_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=True, transforms=tf_train, split_file=split_file)
valid_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=False, transforms=tf_valid, split_file=split_file)


# dataloader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=args.valid_batch_size,
    shuffle=False,
    num_workers=2,  #0
    drop_last=False
)


# model
# 입력 채널: 3 (RGB 이미지), 출력 클래스 수: 29 (다중 클래스 분할)
# Model selection based on `--model_class`
if args.model_class == 'UNet_3Plus':
    model = UNet_3Plus(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)
elif args.model_class == 'UNet_3Plus_DeepSup':
    model = UNet_3Plus_DeepSup(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)
elif args.model_class == 'UNet_3Plus_DeepSup_CGM':
    model = UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)
else:
    raise ValueError(f"Unsupported model class: {args.model_class}")

#weights 선택 (1차, 2차, 3차 가중치)
#1,2는 dice 점수 기반, 3차는 넓이 기반
if args.class_weights == 1:
    class_weights = torch.tensor([
        1.5, 0.8, 0.8, 1.5, 1.25, 0.8, 0.8, 1.5, 1, 0.8, 0.8, 1.5, 1, 0.8, 0.8, 1.5, 1.25, 0.8, 0.8, 1.5,
        1.5, 1.25, 1.25, 1.25, 1.25, 1.5, 1.7, 0.8, 0.8
    ]).cuda()
elif args.class_weights == 2:
    class_weights = torch.tensor([
        1.75, 0.8, 0.8, 1.75, 1.5, 0.8, 0.8, 1.75, 1.25, 0.8, 0.8, 1.75, 1.25, 0.8, 0.8, 1.75, 1.75, 0.8, 0.8,
        1.75, 2.5, 1.5, 1.75, 1.5, 1.5, 1.75, 2.75, 0.8, 0.8
    ]).cuda()
elif args.class_weights == 3:
    class_weights = torch.tensor([
        8.87, 4.25, 2.59, 13.45, 6.86, 3.18, 1.93, 11.45, 5.39, 2.85, 2.14, 11.58, 6.16, 3.37, 2.84, 16.94,
        10.93, 4.79, 2.81, 7.64, 12.60, 4.52, 6.07, 6.50, 8.76, 9.23, 16.02, 1.00, 1.92
    ]).cuda()

#(1, C, 1, 1)로 브로드캐스팅 준비
class_weights = class_weights.view(1, len(CLASSES), 1, 1)

# Loss function 선택
if args.loss_function == 'bce':
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
elif args.loss_function == 'bce+iou+ssim':
    criterion = BCEWithIoUAndSSIM()
elif args.loss_function == 'focal+iou+ssim':
    criterion = FocalLossWithIoUAndSSIM()
elif args.loss_function == 'tversky':
    criterion = TverskyLoss(alpha=0.5, beta=0.5, smooth=1e-6)
elif args.loss_function == 'bce+focal':
    criterion = partial(combined_bce_focal_loss, class_weights=class_weights, focal_weight=0.5)
elif args.loss_function == 'bce+dice':
    criterion = partial(combined_bce_dice, class_weights=class_weights, bce_weight=0.5)
else:
    raise ValueError(f"Unsupported loss function: {args.loss_function}")


# Optimizer
if args.optimizer == 'adam':
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

# 시드를 설정합니다.
set_seed()

train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.val_every, args.saved_dir, args.model_name)

wandb.finish()