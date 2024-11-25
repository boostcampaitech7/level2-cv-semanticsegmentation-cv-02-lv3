import torch
import os
from torch.nn import functional as F
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
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

model1 = AutoModelForSemanticSegmentation.from_pretrained('model_path1')
model2 = torch.load(os.path.join("model_path2"))
model3 = torch.load(os.path.join("model_path3"))
models=[model1,model2,model3]

# 테스트 이미지 경로
image_dir='image_path'

# 저장할 경로
output_dir='output_path'

# 저장할 이름
output_name='output_name'

threshold=0.5

# 모델별 가중치 정의
weights = [1/3, 1/3, 1/3]  # 앙상블 모델의 가중치
# weights=[1]



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

def tta_flip_inference(models, data_loader, weights, thr=0.5):
    """
    Horizontal Flip TTA 기반 테스트 함수

    Args:
        models (list): 앙상블에 사용할 모델 리스트
        data_loader (DataLoader): 테스트 데이터 로더
        weights (list): 모델별 가중치
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
    image_processor = AutoImageProcessor.from_pretrained('image_path1',do_rescale=False)

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input = image_processor(images)
            input = torch.from_numpy(input['pixel_values'][0]).unsqueeze(0).to('cuda')
            images = images.cuda()

            # 모든 모델의 출력을 저장
            all_model_outputs = []
            for i, model in enumerate(models):
                model_outputs = []

                # 원본 이미지 추론
                if i==0:
                    outputs = model(input)
                    outputs = outputs['logits']
                else:
                    outputs = model(images)
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
                model_outputs.append(outputs)  # logits 그대로 저장

                # Horizontal Flip 이미지 추론
                flipped_images = torch.flip(images, dims=[3])  # 좌우 반전
                if i==0:
                    flipped_input=image_processor(flipped_images)
                    flipped_input=flipped_input['pixel_values'][0].unsqueeze(0).to('cuda')
                    outputs_flipped = model(flipped_input)
                    outputs_flipped = outputs_flipped['logits']
                else:
                    outputs_flipped = model(flipped_images)
                outputs_flipped = F.interpolate(outputs_flipped, size=(2048, 2048), mode="bilinear", align_corners=False)
                outputs_flipped = torch.flip(outputs_flipped, dims=[3])  # 다시 원래 방향으로 복원
                model_outputs.append(outputs_flipped)  # logits 그대로 저장

                # TTA 결과 결합
                tta_outputs = torch.mean(torch.stack(model_outputs), dim=0)  # 평균 낸 logits
                tta_outputs = torch.sigmoid(tta_outputs)  # 평균 후 sigmoid 적용

                all_model_outputs.append(tta_outputs * weights[i])

            # 모델별 결과 결합
            weighted_avg_outputs = torch.sum(torch.stack(all_model_outputs), dim=0)

            # 임계값 적용하여 최종 마스크 생성
            final_masks = (weighted_avg_outputs > thr).detach().cpu().numpy()

            # RLE 인코딩 및 저장
            for output, image_name in zip(final_masks, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

    return rles, filename_and_class
tf = A.Resize(1024, 1024)
test_dataset = XRayInferenceDataset(image_root=image_dir,transforms=tf)


test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False

)
# TTA 기반 추론
rles, filename_and_class = tta_flip_inference(models, test_loader,weights, threshold)

# 결과 저장
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})
df.to_csv(output_dir + output_name, index=False)