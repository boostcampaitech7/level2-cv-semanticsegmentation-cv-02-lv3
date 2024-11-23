import numpy as np
import pydensecrf.densecrf as dcrf
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from dataset import XRayInferenceDataset
from tqdm.auto import tqdm
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import albumentations as A
import os

model_dir='/data/ephemeral/home/Seungcheol/ensemble/best_model/hrnet2_2_batch_3_final_1024_best_model.pt'

# 테스트 이미지 경로
image_dir='/data/ephemeral/home/data/test/DCM'

output_dir='/data/ephemeral/home/Seungcheol/ensemble/crf_output'

# 저장할 이름
output_name='hrnetv2_crf.csv'

def apply_crf(image, logits, classes):
    """
    CRF 후처리 함수

    Args:
        image (numpy.ndarray): 원본 이미지 (H x W x C)
        logits (numpy.ndarray): 모델의 출력 (C x H x W) - softmax를 적용한 확률값
        classes (int): 클래스 개수

    Returns:
        numpy.ndarray: CRF가 적용된 결과 (H x W)
    """
    # DenseCRF 객체 생성
    height, width = image.shape[:2]
    d = dcrf.DenseCRF2D(width, height, classes)

    # Unary potentials 추가 (모델 출력 확률값 -> unary 에너지로 변환)
    unary = unary_from_softmax(logits)
    d.setUnaryEnergy(unary)

    # Pairwise potentials 추가 (공간적 연속성을 위한 필터)
    # 1. Gaussian pairwise potential (픽셀 위치 기반 연속성)
    d.addPairwiseGaussian(sxy=3, compat=3)

    # 2. Bilateral pairwise potential (픽셀 위치 + 색상 기반 연속성)
    d.addPairwiseBilateral(sxy=20, srgb=13, rgbim=image, compat=10)

    # Inference 수행
    Q = d.inference(5)  # 5번의 CRF 반복 수행
    result = np.argmax(Q, axis=0).reshape((height, width))

    return result

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

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)  #['out']
            
            logits=outputs.detach().cpu().numpy()
            logits = F.interpolate(logits, size=(2048, 2048), mode="bilinear")
            probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
            image = images.cpu().numpy().transpose(0, 2, 3, 1)
            crf_result = apply_crf(image, probabilities, classes=len(CLASSES))
            
            for output, image_name in zip(crf_result, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class



model = torch.load(model_dir)

tf = A.Resize(1024, 1024)
test_dataset = XRayInferenceDataset(image_root=image_dir,transforms=tf)


test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1,
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

df.to_csv(output_dir+output_name, index=False)


# # 모델 결과 얻기 (logits: C x H x W)
# logits = model_output["logits"].detach().cpu().numpy()

# # softmax를 적용해 클래스별 확률값 계산
# probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()


# # 데이터 로더에서 원본 이미지 가져오기
# image = original_image.cpu().numpy().transpose(1, 2, 0)  # H x W x C

# # CRF 적용
# crf_result = apply_crf(image, probabilities, classes=len(CLASSES))

# # CRF 결과를 RLE로 변환
# crf_rle = encode_mask_to_rle(crf_result)