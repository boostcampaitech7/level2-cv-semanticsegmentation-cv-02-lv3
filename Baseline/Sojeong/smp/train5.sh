python train.py \
    --epochs 80 \
    --batch_size 4 \
    --valid_batch_size 4 \
    --model_name tu-resnext101_32x16d \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model DeepLabV3Plus \

python inference.py \
    --model_name tu-resnext101_32x16d7 \
    --seg_model DeepLabV3Plus \


# chmod +x train5.sh
# ./train5.sh

#tu-tf_efficientnet_b7