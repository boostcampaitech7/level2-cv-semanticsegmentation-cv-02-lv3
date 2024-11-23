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
# 모델 리스트 정의
model1 = AutoModelForSemanticSegmentation.from_pretrained('/data/ephemeral/home/Seungcheol/ensemble/ep_47_vdice_0.9716')
model2 = torch.load(os.path.join("/data/ephemeral/home/Seungcheol/ensemble/best_model/hrnet2_2_batch_3_final_1024_best_model.pt"))
model3 = torch.load(os.path.join("/data/ephemeral/home/Seungcheol/ensemble/best_model/UnetPlusPlus_tu-hrnet_w48_[1024, 1024]_batch2_fold0__hybrid2_best_model.pt"))
models = [model1, model2, model3]

# 각 모델에 주고 싶은 가중치
w1=1/3
w2=1/3
w3=1/3

# 1로 결정지고 싶은 확률 기준 
threshold=0.5

# 테스트 이미지 경로
image_dir='/data/ephemeral/home/data/test/DCM'

# 저장할 경로
output_dir='/data/ephemeral/home/Seungcheol/ensemble/soft_voting/'

# 저장할 이름
output_name='hrnetv2_upernet_0.9716_unet++(aug+grid).csv'




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

def test(models, data_loader, thr=0.5):
    """
    Soft Voting 기반 테스트 함수

    Args:
        models (list): 앙상블에 사용할 모델 리스트
        data_loader (DataLoader): 테스트 데이터 로더
        thr (float): 최종 확률 임계값 (기본값 0.5)

    Returns:
        rles (list): RLE 형태의 결과
        filename_and_class (list): 파일명과 클래스 리스트
    """
    for model in models:
        model = model.cuda()
        model.eval()

    rles = []
    filename_and_class = []
    image_processor = AutoImageProcessor.from_pretrained('/data/ephemeral/home/Seungcheol/ensemble/ep_47_vdice_0.9716',do_rescale=False)
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input = image_processor(images)
            input = torch.from_numpy(input['pixel_values'][0]).unsqueeze(0).to('cuda')
            images = images.cuda()

            # 모든 모델의 출력을 저장
            all_outputs = []
            for i,model in enumerate(models):
                if i==0:
                    outputs = model(input)
                    outputs = outputs['logits']
                else:
                    outputs = model(images)
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)  # 확률값으로 변환
                all_outputs.append(outputs)

            # 모델별 가중치
            weights = [w1 ,w2, w3]  # 가중치 합은 1로 설정

            # 가중 평균 계산
            weighted_avg_outputs = torch.sum(
                torch.stack(all_outputs) * torch.tensor(weights, device="cuda").view(-1, 1, 1, 1), 
                dim=0
            )

            # 임계값 적용하여 최종 마스크 생성
            final_masks = (weighted_avg_outputs > thr).detach().cpu().numpy()

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

rles, filename_and_class = test(models, test_loader, threshold)
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(output_dir+output_name, index=False)