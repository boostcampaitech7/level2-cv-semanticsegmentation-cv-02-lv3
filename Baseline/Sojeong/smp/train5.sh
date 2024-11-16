python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name timm-efficientnet-b7 \
    --val_every 10 \
    --encoder_weights noisy-student \
    --seg_model UnetPlusPlus \

python inference.py \
    --model_name timm-efficientnet-b7 \
    --seg_model UnetPlusPlus \


# chmod +x train5.sh
# ./train5.sh

#tu-tf_efficientnet_b7