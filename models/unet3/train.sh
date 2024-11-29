#!/bin/bash

# Training configuration
EPOCHS=50                      # 학습 에포크 수
LR=0.001                       # Learning rate
BATCH_SIZE=2                   # 배치 크기
VALID_BATCH_SIZE=1             # 검증 데이터 배치 크기
VAL_EVERY=1                    # Validation 주기
IMAGE_ROOT="/data/ephemeral/home/data/train/DCM"     # 이미지 데이터 경로
LABEL_ROOT="/data/ephemeral/home/data/train/outputs_json"     # 레이블 데이터 경로
SAVED_DIR="checkpoints"         # 모델 체크포인트 저장 경로
MODEL_FILE=${1:-"resnext"}              # model 파일 선택 (base, resnext, swin_t)
MODEL_CLASS=${2:-"UNet3Plus_ResNeXt101"} # model 클래스 선택
LOSS_FUNCTION="bce+dice"        # 손실 함수 선택
OPTIMIZER="adam"                # 옵티마이저 선택
CLASS_WEIGHTS=${3:-3}           # 기본 클래스 가중치 (3)
IMAGE_SIZE=${4:-512}            # 기본 이미지 크기
AUGMENTATIONS=${5:-""}          # 기본 augmentations 없음

# Run training
python train.py \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --valid_batch_size $VALID_BATCH_SIZE \
    --val_every $VAL_EVERY \
    --image_root $IMAGE_ROOT \
    --label_root $LABEL_ROOT \
    --saved_dir $SAVED_DIR \
    --model_file $MODEL_FILE \
    --model_class $MODEL_CLASS \
    --loss_function $LOSS_FUNCTION \
    --optimizer $OPTIMIZER \
    --class_weights $CLASS_WEIGHTS \
    --image_size $IMAGE_SIZE \
    --augmentations "$AUGMENTATIONS"
