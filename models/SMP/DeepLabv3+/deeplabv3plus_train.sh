python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-xception71 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model DeepLabV3Plus \
    --resize 1024 1024 \
    --image_root  \
    --label_root  \
    --json_dir  \
    --fold 0 \