
python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 1 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --train_batch 2 \
    --fold 1 \

python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 2 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --train_batch 2 \
    --fold 2 \

python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 3 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --train_batch 2 \
    --fold 3 \

python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --image_root /data/ephemeral/home/data/train/DCM \
    --label_root /data/ephemeral/home/data/train/outputs_json \
    --json_dir /data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Data/train_valid_split/splits \
    --fold 4 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --train_batch 2 \
    --fold 4 \
# chmod +x train.sh
# ./train.sh