# 동작 방법

## 기본
bash train.sh 

## 가중치 사용 (1차, 2차, 3차 가중치 중 택)
bash train.sh 3

## 가중치, 이미지 사이즈 변경
bash train.sh 512 3

## 가중치, 이미지 사이즈, augmentation 변경 
bash train.sh 3 512 "resize grid brightness"

## encoder 변경
bash train.sh resnext UNet3Plus_ResNeXt101 3 512

bash train.sh swin_t UNet3Plus_Swin 3 616

## loss function, optimizer도 변경 가능함.
