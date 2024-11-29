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
    parser.add_argument('--model_class', type=str, default='UNet_3Plus', 
                        choices=['UNet_3Plus', 'UNet_3Plus_DeepSup', 'UNet_3Plus_DeepSup_CGM'], 
                        help='Model class to use')
    parser.add_argument('--loss_function', type=str, default='bce', 
                        choices=['bce', 'combined'], 
                        help='Loss function to use: bce or combined')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'rmsprop'], 
                        help='Optimizer to use: adam or rmsprop')
    
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

# dataset
tf = A.Resize(512, 512)
train_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=True, transforms=tf)
valid_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=False, transforms=tf)

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

# Loss function 선택
if args.loss_function == 'bce':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss_function == 'combined':
    criterion = combined_loss_with_dynamic_weights
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