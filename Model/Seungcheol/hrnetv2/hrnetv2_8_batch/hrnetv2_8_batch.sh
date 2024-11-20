python train.py \
    --epochs 100 \
    --batch_size 8 \
    --valid_batch_size 2 \
    --model_name hrnetv2 \
    --val_every 10 \

python inference.py \
    --model_name hrnetv2\

# chmod +x hrnetv2_8_batch.sh
# ./hrnetv2_8_batch.sh