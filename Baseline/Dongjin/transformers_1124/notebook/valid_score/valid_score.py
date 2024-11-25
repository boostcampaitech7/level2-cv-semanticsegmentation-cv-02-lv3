import sys
sys.path.append('../../')

from glob import glob
import pandas as pd
import os
import utils
from train import dice_coef
from inference import decode_rle_to_mask, get_xray_classes
import numpy as np
import cv2
import torch
from tqdm import tqdm

def find_target_json_path(image_name, json_paths):
    image_name_ = os.path.splitext(image_name)[0]
    target_json_paths = [json_path for json_path in json_paths if image_name_ in json_path]

    if len(target_json_paths) != 1:
        raise(Exception(f'len(target_json_paths) is {len(target_json_paths)}'))

    target_json_path = target_json_paths[0]
    return target_json_path

def get_gt_label(anns, classes, image_size):   # (H, W, NC) 모양의 label을 생성
    label_shape = (classes['num_class'], ) + image_size 
    label = np.zeros(label_shape, dtype=np.uint8)

    # 클래스 별로 처리
    for ann in anns:
        class_name = ann["label"]
        class_ind = classes['class2idx'][class_name]
        points = np.array(ann["points"])

        class_label = np.zeros(image_size, dtype=np.uint8)
        cv2.fillPoly(class_label, [points], 1)
        label[class_ind, :, :] = class_label

    return label

def get_valid_score(current_dir_path, predict_path):
    save_name = os.path.basename(predict_path).replace('.csv', '.json')
    save_dir_path = os.path.join(current_dir_path, 'result')
    save_path = os.path.join(save_dir_path, save_name)
    os.makedirs(save_dir_path, exist_ok=True)

    if os.path.exists(save_path):
        print(f'{save_path} is already existed')
        return 
    else:
        with open(save_path, 'w') as f:
            pass

    data_dir_path = '/data/ephemeral/home/data'
    image_size = (2048, 2048)

    json_paths = glob(data_dir_path + '/**/*.json', recursive=True)
    df = pd.read_csv(predict_path)
    image_names = df['image_name'].unique().tolist()
    n_images = len(image_names)
    classes = get_xray_classes()

    print(f"n_images: {n_images}")

    if n_images != 160:
        raise(Exception("Check n_images: {n_images}"))

    results = []

    for image_name in tqdm(image_names):
        target_json_path = find_target_json_path(image_name, json_paths)
        anns = utils.read_json(target_json_path)['annotations']
        gt = get_gt_label(anns, classes, image_size)

        df_image_name = df[df['image_name'] == image_name]
        rles = df_image_name['rle'].values.tolist()

        preds = []
        for rle in rles:
            pred = decode_rle_to_mask(rle, height=image_size[0], width=image_size[1])
            preds.append(pred)

        preds = np.stack(preds, 0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        preds = torch.from_numpy(preds).unsqueeze(0)

        result = dice_coef(gt, preds)
        results.append(result)

    results = torch.concat(results)

    dicts = {}
    dicts['total_dices_mean'] = results.mean(dim=0).mean().item()
    dicts['dices_mean'] = results.mean(dim=0).tolist()
    dicts['dices_std'] = results.std(dim=0).tolist()
    dicts['results'] = results.tolist()

    print(f"{save_name}: total_dices_mean: {dicts['total_dices_mean']:.4f}")
    utils.save_json(dicts, save_path)

if __name__ == '__main__':
    work_dir_path = os.path.dirname(os.path.realpath(__file__))

    predict_paths = ["/data/ephemeral/home/Dongjin/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Dongjin/transformers_1124/ensemble/result/1125_valid_upernet-convnext-small-fold0.csv"]

    for predict_path in predict_paths:
        get_valid_score(work_dir_path, predict_path)