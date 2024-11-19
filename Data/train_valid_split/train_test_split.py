# python native
import os
import json
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

# 데이터 경로를 입력하세요

IMAGE_ROOT = "IMAGE_PATH"
LABEL_ROOT = "LABEL_PATH"



CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}
pngs = sorted(pngs)
jsons = sorted(jsons)



import json
from sklearn.model_selection import GroupKFold

# GroupKFold split 저장 함수
def save_kfold_splits(filenames, labelnames, groups, n_splits=5, output_dir='splits'):
    gkf = GroupKFold(n_splits=n_splits)
    
    for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(filenames, [0] * len(filenames), groups)):
        train_filenames = filenames[train_idx].tolist()
        valid_filenames = filenames[valid_idx].tolist()
        train_labelnames = labelnames[train_idx].tolist()
        valid_labelnames = labelnames[valid_idx].tolist()
        
        split_data = {
            'train_filenames': train_filenames,
            'train_labelnames': train_labelnames,
            'valid_filenames': valid_filenames,
            'valid_labelnames': valid_labelnames
        }
        
        # JSON 파일로 저장
        with open(f'{output_dir}/fold_{fold_idx}.json', 'w') as f:
            json.dump(split_data, f)

# 데이터 초기화 및 GroupKFold split 저장
filenames = np.array(pngs)
labelnames = np.array(jsons)
groups = [os.path.dirname(fname) for fname in filenames]

save_kfold_splits(filenames, labelnames, groups)


# 폴더와 fold 수 설정
output_dir = 'splits'
n_splits = 5

# 각 fold의 train과 validation 파일 리스트를 담을 집합 초기화
all_valid_files = set()

# 각 fold 내의 train과 validation 간 교집합 확인 및 모든 validation 파일 수집
for fold_idx in range(n_splits):
    split_file = os.path.join(output_dir, f'fold_{fold_idx}.json')
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    train_files = set(split_data['train_filenames'])
    valid_files = set(split_data['valid_filenames'])
    
    # 각 fold 내의 교집합 확인
    intersection = train_files & valid_files
    if intersection:
        print(f"Fold {fold_idx} has overlapping files between train and valid sets:", intersection)
    else:
        print(f"Fold {fold_idx} has no overlapping files between train and valid sets:")
    
    # 모든 fold의 validation 파일 수집
    all_valid_files.update(valid_files)

# 모든 fold의 validation 파일 간 중복 확인
valid_intersection = set()
valid_files_list = list(all_valid_files)

for i in range(len(valid_files_list)):
    for j in range(i + 1, len(valid_files_list)):
        if valid_files_list[i] == valid_files_list[j]:
            valid_intersection.add(valid_files_list[i])

if valid_intersection:
    print("Overlapping files found among all validation sets:", valid_intersection)
else:
    print("No overlapping files found among all validation sets.")

