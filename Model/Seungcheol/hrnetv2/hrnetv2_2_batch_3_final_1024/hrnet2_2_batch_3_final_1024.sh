python train.py \
    --epochs 100 \
    --batch_size 2 \
    --valid_batch_size 2 \
    --model_name hrnet2_2_batch_3_final_1024\
    --val_every 10 \
    --image_resize 1024\

python inference.py \
    --model_name hrnet2_2_batch_3_final_1024\
    --image_resize 1024\

# chmod +x hrnet2_2_batch_3_final_1024.sh
# ./hrnet2_2_batch_3_final_1024.sh