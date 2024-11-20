python train.py \
    --epochs 100 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name hrnet2_2_batch_3_final\
    --val_every 10 \

python inference.py \
    --model_name hrnet2_2_batch_3_final\

# chmod +x hrnet2_2_batch_3_final.sh
# ./hrnet2_2_batch_3_final.sh