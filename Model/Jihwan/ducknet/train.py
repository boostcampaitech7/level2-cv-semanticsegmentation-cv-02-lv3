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
from loss import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='valid Batch size')
    parser.add_argument('--val_every', type=int, default=1, help='val_every')
    parser.add_argument('--image_root', type=str,default='/data/ephemeral/home/data/train/DCM',help='image root')
    parser.add_argument('--label_root',type=str,default='/data/ephemeral/home/data/train/outputs_json',help='label root')
    parser.add_argument('--saved_dir', type=str, default='/data/ephemeral/home/Jihwan/ducknet/checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='ducknet', help='Name of the segmentation model')
    parser.add_argument('--json_dir', type=str, default='/data/ephemeral/home/Jihwan/split', help='train_valid_split_dir')
    parser.add_argument('--fold', type=int,default=0,help='split_k_fold_0')
    args = parser.parse_args()

    
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

RANDOM_SEED = 42

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



# 시드를 설정합니다.
set_seed()

# json 파일 경로 지정
split_file = args.json_dir+f'/fold_{args.fold}.json'

# dataset
tf = A.Resize(1024, 1024)
train_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=True, transforms=tf,split_file=split_file)
valid_dataset = XRayDataset(image_root=args.image_root, label_root=args.label_root, is_train=False, transforms=tf,split_file=split_file)

# dataloader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)


# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=args.valid_batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=False
)


# model
# 입력 채널: 3 (RGB 이미지), 출력 클래스 수: 29 (다중 클래스 분할)
#model = UNet_3Plus_DeepSup(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)

model = DUCKNet(img_height=1024, img_width=1024, input_channels=3, out_classes=29, starting_filters=34)
# model = torch.load("/data/ephemeral/home/Jihwan/ducknet/model.pth")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print('---------------------------')

class_weights = torch.tensor([
    1.750, 0.800, 0.800, 1.750, 1.500, 0.800, 0.800, 1.750, 1.250, 0.800, 
    0.800, 1.750, 1.250, 0.800, 0.800, 1.750, 1.750, 0.800, 0.800, 1.750, 
    2.500, 1.500, 1.750, 1.500, 1.500, 1.750, 2.750, 0.800, 0.800
]).cuda()

class_weights = class_weights.view(1, len(CLASSES), 1, 1) # (1, C, 1, 1)로 브로드캐스팅 준비


def criterion(pred, label):
    return combined_bce_dice_loss(pred, label, class_weights, smooth=1e-6)

# Loss function을 정의합니다.
loss = criterion

# Optimizer를 정의합니다.
optimizer = optim.RMSprop(params=model.parameters(), lr=args.lr, weight_decay=1e-6)


train(model, train_loader, valid_loader, loss, optimizer, args.epochs, args.val_every, args.saved_dir, args.model_name)


wandb.finish()