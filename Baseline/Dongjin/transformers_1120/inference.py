from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
import os
import utils
from dataset import XRayDataset, get_xray_classes
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


class Inference:
    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self.conf_path = os.path.join(self.model_dir_path , 'exp.json')
        self.conf = utils.read_json(self.conf_path)
        self.conf['step_batch_size'] = 1
        
        # 실재 모델이 저장된 위치를 찾아 image_processor 및 model 불러오기
        self.saved_model_dir_path = utils.get_saved_model_dir_path(self.model_dir_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.saved_model_dir_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.saved_model_dir_path).to(self.conf['device'])

        self.load_dataset_and_loader()
        self.classes = get_xray_classes()


    def load_dataset_and_loader(self):
        self.train_dataset = XRayDataset(mode='train', 
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['train_json_path'])

        self.valid_dataset = XRayDataset(mode='valid', 
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['valid_json_path'])

        self.test_dataset = XRayDataset(mode='test', 
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['test_json_path'])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf['step_batch_size'], shuffle=True, num_workers=self.conf['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.conf['step_batch_size'], shuffle=False, num_workers=self.conf['num_workers'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.conf['step_batch_size'], shuffle=False, num_workers=self.conf['num_workers'])

    
    def inference(self, mode, thr):
        if mode == 'train':
            data_loader = self.train_dataset
        elif mode == 'valid':
            data_loader = self.valid_loader
        elif mode == 'test':
            data_loader = self.test_loader
        else:
            raise(Exception("mode supports train/valid/test"))

        rles = []
        filename_and_class = []

        self.model.eval()
        with torch.no_grad():
            for batch, image_names in tqdm(data_loader, total=len(data_loader)):
                inputs = batch["pixel_values"].to('cuda')
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(0)
                
                outputs = self.model(inputs).logits
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{self.classes['idx2class'][c]}_{image_name}")
                        
        return rles, filename_and_class
    
    def inference_and_save(self, mode, save_path=None, thr=0.5):
        rles, filename_and_class = self.inference(mode, thr)
        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]

        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })

        if save_path is None:
            prefix = os.path.basename(self.saved_model_dir_path)
            suffix = os.path.basename(self.model_dir_path)
            save_name = f'{mode}_{prefix}_{suffix}.csv'
            save_path = os.path.join(self.model_dir_path, save_name)

        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    model_dir_path = '/data/ephemeral/home/Dongjin/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Dongjin/transformers_1120/trained_models/cont/upernet-convnext-small_cont_size_1024_cont_weight'
    inference = Inference(model_dir_path)
    inference.inference_and_save(mode='train')
    inference.inference_and_save(mode='valid')
