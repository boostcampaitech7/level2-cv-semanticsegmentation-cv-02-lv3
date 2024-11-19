from ultralytics import YOLO
import torch
import json
import csv
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ultralytics.engine.results import Masks
import os

def encode_mask_to_rle(mask_data):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask_data.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Load a model
model = YOLO("/data/ephemeral/home/Jungyeon/Yolo/runs/segment/train2/weights/best.pt")  # load a custom model

test_data = "/data/ephemeral/home/data/test/DCM"
  
rles = []
class_name = []
class_num = []
image_name = []
for root, dirs, files in os.walk(test_data):
    dirs.sort()
    files.sort()
    for file_name in files:
        print(len(class_name))
        print(len(class_num))
        file_path = os.path.join(root, file_name)
        # Predict with the model
        results = model.predict(file_path)  # predict on an image
        
        for result in results:
            df_result = result.to_df()

        class_name += df_result['name'].tolist()
        class_num += df_result['class'].tolist()
        image_name += [file_name]*len(df_result['name'])
        masks = results[0].masks.data.cpu()

        for mask in masks:
            rle = encode_mask_to_rle(mask)
            rles.append(rle)
   
  
df = pd.DataFrame({
    "image_name": image_name,
    "class": class_name,
    "class_num" : class_num,
    "rle": rles,
})

# 클래스 기준 오름차순
df = (
    df.groupby("image_name", group_keys=False)  # groupby 설정
    .apply(lambda x: x.sort_values("class_num"))  # 그룹별 정렬
)
# 'class_num' 열 삭제
df = df.drop(columns=["class_num"])
# 중복된 클래스 에측값 중 첫번째 값 제외 제거
df = df.drop_duplicates(subset=['image_name', 'class'], keep='first')
# csv 파일로 저장
df.to_csv(f"df_final_sorted_class_num.csv", index=False)