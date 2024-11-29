# 5-Fold Cross Validation
for fold in {1..4}  # fold를 0부터 시작 (5-fold에서는 0~4)
do
    echo "Starting training for fold ${fold}..."
    python train.py \
        --epochs 60 \
        --model_name "${fold}_fold" \
        --fold ${fold}  # fold 값 전달
    
    echo "Training for fold ${fold} completed. Starting inference..."
    python inference.py \
        --model_name "${fold}_fold"\
        --batch_size 2            

    echo "Inference for fold ${fold} completed."
done


# chmod +x last.sh