# python train.py \
#     --epochs 100 \
#     --batch_size 2 \
#     --valid_batch_size 2 \
#     --model_name efficientnet-b7 \
#     --val_every 10 \
#     --encoder_weights imagenet \

python inference.py \
    --model_name efficientnet-b7 \

# chmod +x train2.sh
# ./train2.sh