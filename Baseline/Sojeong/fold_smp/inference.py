# python native
import os
import json
import random
import datetime
from functools import partial

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

from dataset import XRayInferenceDataset
from trainer import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--image_root', type=str,default='/data/ephemeral/home/data/test/DCM',help='image root')
    parser.add_argument('--saved_dir', type=str, default='checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='resnet101', help='model name')
    parser.add_argument('--batch_size',type=int,default=8,help='batch_size')
    parser.add_argument('--seg_model', type=str, default='UnetPlusPlus', help='Segmentation model name')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help='Resize dimensions: height width')
    parser.add_argument('--train_batch', type=int, default=8, help='Train batch size')
    parser.add_argument('--fold', type=int, default=0, help='kfold')
    
    args = parser.parse_args()

if not os.path.exists('output'):                                                           
    os.makedirs('output')

model = torch.load(os.path.join(args.saved_dir, f"{args.seg_model}_{args.model_name}_{args.resize}_batch{args.train_batch}_fold{args.fold}__hybrid2_best_model.pt"))

tf = A.Resize(args.resize[0], args.resize[1])

test_dataset = XRayInferenceDataset(image_root=args.image_root,transforms=tf)


test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test(model, test_loader)

classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(f"output/{args.seg_model}_{args.model_name}_{args.resize}_batch{args.train_batch}_fold{args.fold}__hybrid2_output.csv", index=False)

# python inference.py