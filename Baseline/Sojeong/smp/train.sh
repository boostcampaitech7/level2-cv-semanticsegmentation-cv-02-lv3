python train.py \
    --epochs 80 \
    --batch_size 1 \
    --valid_batch_size 1 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 768 768 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 768 768 \
    --train_batch 1 \
# chmod +x train.sh
# ./train.sh

#Unet
#Unet++
#FPN
#PSPNet
#DeepLabV3
#DeepLabV3Plus
#Linknet
#MAnet
#PAN
#UPerNet

# efficientnet-b7
# tu-xception71
# tu-efficientnet_b7
# resnext101_32x8d
# timm-efficientnet-b7 (noisy-student weight)
# tu-resnext101_32x16d