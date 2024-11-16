python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-resnext101_32x16d \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \

python inference.py \
    --model_name tu-resnext101_32x16d \
    --seg_model UnetPlusPlus \

# chmod +x train3.sh
# ./train3.sh

# 이후에 
# efficientnet-b4
# efficientnet-b7
# timm-efficientnet-b7 #  noisy-student
# tu-coatnet_3_rw_224
# tu-xception71
# tu-tf_efficientnet_l2
# tu-tf_efficientnet_b7
# tu-efficientnet_b7
# maxvit_large_tf_384
# mobilenetv3_large_100
# resnext101_32x8d
# resnext101_32x16d

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