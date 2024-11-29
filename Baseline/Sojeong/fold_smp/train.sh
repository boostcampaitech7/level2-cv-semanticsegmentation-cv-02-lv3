
python train.py \
    --epochs 60 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-hrnet_w48 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --image_root  \
    --label_root  \
    --json_dir  \
    --fold 1 \

python inference.py \
    --model_name tu-hrnet_w48 \
    --seg_model UnetPlusPlus \
    --resize 1024 1024 \
    --train_batch 2 \
    --fold 1 \

