# python train.py \
#     --epochs 100 \
#     --batch_size 2 \
#     --valid_batch_size 2 \
#     --model_name resnext101_32x8d \
#     --val_every 10 \
#     --encoder_weights imagenet \

python inference.py \
    --model_name resnext101_32x8d \

# chmod +x train.sh
# ./train.sh