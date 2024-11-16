python train2.py \
    --epochs 80 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name tu-xception71 \
    --val_every 10 \
    --encoder_weights imagenet \
    --seg_model UnetPlusPlus \
    --decoder_attention_type scse \
    --activation None \

python inference.py \
    --model_name tu-xception71 \
    --seg_model UnetPlusPlus \

# chmod +x train4.sh
# ./train4.sh