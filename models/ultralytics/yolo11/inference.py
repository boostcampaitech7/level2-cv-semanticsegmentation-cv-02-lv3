import json
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

def encode_mask_to_rle(mask_data):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask_data.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_sorted_file_paths(data_path):
    images = []

    for root, dirs, files in os.walk(data_path):
        dirs.sort()
        files.sort()

        images.extend(os.path.join(root, file_name) for file_name in files)

    return images

def df_to_csv(dataframe, output_file):
    dataframe = (
        dataframe.sort_values(by=["image_name", "class_num"])  # 정렬
        .drop(columns=["class_num"])                           # 불필요한 열 제거
        .drop_duplicates(subset=["image_name", "class"])       # 중복 제거
    )
    
    # CSV 파일로 저장
    dataframe.to_csv(output_file, index=False)

def inference(pretrained_weight, data_path):
    results = []

    model = YOLO(pretrained_weight)
    test_images = get_sorted_file_paths(data_path)
    outputs = model.predict(test_images)  # 예측 수행

    for file_name, output in tqdm(zip(test_images, outputs), total=len(test_images), desc="Processing Images"):
        df_result = output.to_df()  # DataFrame으로 변환

        class_names = df_result['name'].tolist()
        class_index = df_result['class'].tolist()
        masks = output.masks.data.cpu()

        for i, mask in enumerate(masks):
            rle = encode_mask_to_rle(mask)
            results.append({
                "image_name": file_name,
                "class": class_names[i],
                "class_num": class_indices[i],
                "rle": rle
            })
            
    df = pd.DataFrame(results)
    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a YOLO model and save results.")
    parser.add_argument("--pretrained_weight", type=str, required=True, help="Path to the pretrained YOLO model weights.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing test images.")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to resulting CSV file (default: submission.csv).")
    args = parser.parse_args()

    # Perform inference and save results
    df = inference(args.pretrained_weight, args.data_path)
    df_to_csv(df, args.output_csv)
    
    print(f"Inference completed and results saved to {args.output_csv}")