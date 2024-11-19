python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 0 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --train_batch 2 \
    --fold 0 \

python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 1 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --train_batch 2 \
    --fold 1 \


python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 2 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --train_batch 2 \
    --fold 2 \

python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 3 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --train_batch 2 \
    --fold 3 \

python train.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 4 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 512 512 \
    --train_batch 2 \
    --fold 4 \

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