python train.py \
    --epochs 100 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name hrnetv2_batch_2\
    --val_every 10 \

python inference.py \
    --model_name hrnetv2_batch_2\

# chmod +x hrnetv2_2_batch.sh
# ./hrnetv2_2_batch.sh