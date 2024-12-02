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
from dataset import XRayInferenceDataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
# 사용자 정의 투표 기준 사용
voting_threshold = 2  # 최소 2개의 모델이 동일하게 1로 예측해야 1로 설정

# 1로 결정지고 싶은 확률 기준 
thr1=0.5
thr2=0.5
thr3=0.5

# 모델 리스트 정의
# transformers 모델 로드할 때
model1 = AutoModelForSemanticSegmentation.from_pretrained('model_path1')

# smp,hrnetv2,torchvision 로드 할 때
model2 = torch.load(os.path.join("model_path2"))
model3 = torch.load(os.path.join("model_path3"))
models = [(model1,thr1), (model2,thr2), (model3,thr3)]

# 테스트 이미지 경로
image_dir='image_path'

# 저장할 경로
output_dir='output_path'

# 저장할 이름
output_name='output_name'




CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def test(models, data_loader, voting_threshold=2):
    """
    사용자 정의 투표 기준 기반 테스트 함수

    Args:
        models (list): 앙상블에 사용할 모델 리스트
        data_loader (DataLoader): 테스트 데이터 로더
        thr (float): 각 모델의 Threshold 기준 (기본값 0.5)
        voting_threshold (int): 픽셀 예측값이 1로 설정되기 위한 최소 투표 수

    Returns:
        rles (list): RLE 형태의 결과
        filename_and_class (list): 파일명과 클래스 리스트
    """
    for model,_ in models:
        model = model.cuda()
        model.eval()

    rles = []
    filename_and_class = []
    image_processor = AutoImageProcessor.from_pretrained('model_path1',do_rescale=False)
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input = image_processor(images)
            input = torch.from_numpy(input['pixel_values'][0]).unsqueeze(0).to('cuda')
            images = images.cuda()

            # 모든 모델의 출력을 저장
            binary_outputs = []
            for i,(model,thr) in enumerate(models):
                if i==0:
                    outputs = model(input)
                    outputs = outputs['logits']
                else:
                    outputs = model(images)
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)  # 확률값으로 변환
                outputs = (outputs > thr).float()  # Threshold 기준으로 0과 1로 변환
                binary_outputs.append(outputs)

            # 사용자 정의 투표 기준 적용
            votes = torch.sum(torch.stack(binary_outputs), dim=0)  # 모델별 결과 합산
            final_masks = (votes >= voting_threshold).int().cpu().numpy()  # 사용자 정의 기준으로 최종 결과 생성

            # RLE 인코딩 및 저장
            for output, image_name in zip(final_masks, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class



tf = A.Resize(1024, 1024)
test_dataset = XRayInferenceDataset(image_root=image_dir,transforms=tf)


test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test(models, test_loader, voting_threshold)

classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(output_dir+output_name, index=False)