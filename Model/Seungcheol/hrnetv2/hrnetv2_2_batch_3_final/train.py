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
from hrnet import get_seg_model

from config3 import cfg
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='valid Batch size')
    parser.add_argument('--val_every', type=int, default=10, help='val_every')
    parser.add_argument('--image_root', type=str,default='image_path',help='image root')
    parser.add_argument('--label_root',type=str,default='label_path',help='label root')
    parser.add_argument('--saved_dir', type=str, default='checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='hrnetv2', help='Name of the segmentation model')
    parser.add_argument('--image_resize',type=int,default=512,help='image resize')
    
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

# dataset
tf = A.Resize(args.image_resize, args.image_resize)
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


# # model
# model_func = getattr(models.segmentation, args.model_name)
# model = model_func(pretrained=True)

# # output class 개수를 dataset에 맞도록 수정합니다.
# model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
model=get_seg_model(cfg)


# Loss function을 정의합니다.
criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

# 시드를 설정합니다.
set_seed()

train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.val_every, args.saved_dir, args.model_name)

wandb.finish()
