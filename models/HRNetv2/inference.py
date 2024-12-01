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
    parser.add_argument('--image_root', type=str,default='IMAGE_PATH',help='image root')
    parser.add_argument('--saved_dir', type=str, default='checkpoints',help='model checkpoint save')
    parser.add_argument('--model_name', type=str, default='HRNetv2', help='model name')
    parser.add_argument('--batch_size',type=int,default=2,help='batch_size')
    parser.add_argument('--image_resize',type=int,default=1024,help='image resize')
    
    
    args = parser.parse_args()

if not os.path.exists('output'):                                                           
    os.makedirs('output')

model = torch.load(os.path.join(args.saved_dir, f"{args.model_name}_best_model.pt"))

tf = A.Resize(args.image_resize, args.image_resize)
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

df.to_csv(f"output/{args.model_name}_output.csv", index=False)