#!/bin/bash

# Train Script
python train.py \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 2 \
    --valid_batch_size 1 \
    --val_every 1 \
    --model_class UNet_3Plus \
    --model_name "unet3plus_baseline"
