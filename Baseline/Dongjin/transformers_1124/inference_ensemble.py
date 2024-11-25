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
from argparse import ArgumentParser



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
    def __init__(self, model_dir_path, TTA):
        self.model_dir_path = model_dir_path
        self.TTA = TTA 
        self.conf_path = os.path.join(self.model_dir_path , 'exp.json')
        self.conf = utils.read_json(self.conf_path)
        self.conf['step_batch_size'] = 1
        
        # 실재 모델이 저장된 위치를 찾아 image_processor 및 model 불러오기
        self.saved_model_dir_path = utils.get_saved_model_dir_path(self.model_dir_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.saved_model_dir_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.saved_model_dir_path).to(self.conf['device'])

        self.load_dataset_and_loader()
        self.classes = get_xray_classes(self.conf['crop_type'])


    def load_dataset_and_loader(self):
        crop_info = None
        if self.conf['crop_info_path']:
            crop_info = utils.read_json(self.conf['crop_info_path'])

        self.train_dataset = XRayDataset(mode='train',
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info, 
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['train_json_path'])

        self.valid_dataset = XRayDataset(mode='valid', 
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info,
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['valid_json_path'])

        self.test_dataset = XRayDataset(mode='test', 
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info,
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['test_json_path'])


    def flip_crop(self, crop):
        w, h = [2048, 2048]
        x1, x2, y1, y2 = crop
        x2, x1 = w-x1-1, w-x2-1

        return x1, x2, y1, y2

    def predict(self, inputs, crop):
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
            
        inputs = inputs.to('cuda')
        outputs = self.model(inputs).logits
        
        if crop is None:
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        else:
            INF = 100
            x1, x2, y1, y2 = crop
            w, h = x2-x1+1, y2-y1+1 
            temp = F.interpolate(outputs, size=(h, w), mode="bilinear")
            outputs = -INF * torch.ones(temp.shape[0], temp.shape[1], 2048, 2048).to('cuda')
            outputs[:, :, y1:y2+1, x1:x2+1] = temp
            del temp

        return outputs.sigmoid()


    def inference(self, mode, idx):
        if mode == 'train':
            dataset = self.train_dataset
        elif mode == 'valid':
            dataset = self.valid_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise(Exception("mode supports train/valid/test"))

        batch, image_names = dataset[idx]
        inputs = batch["pixel_values"]
        crop = batch['crop']

        if self.TTA:
            outputs = self.predict(inputs, crop)
            outputs += self.predict(inputs.flip(dims=[-1]), self.flip_crop(crop)).flip(dims=[-1])
            outputs = outputs / 2
        else:
            outputs = self.predict(inputs, crop)

        return outputs.detach().cpu(), image_names


def load_ensemble_conf(work_dir_path, rel_ensemble_conf_path):
    ensemble_conf_path = os.path.join(work_dir_path, rel_ensemble_conf_path)
    conf = utils.read_json(ensemble_conf_path)
    conf['model_dir_paths'] = []

    for model_dir_path_format in conf['model_dir_path_formats']:
        model_dir_path = model_dir_path_format.format(**conf)
        conf['model_dir_paths'].append(model_dir_path)

    if not utils.is_unique(conf['model_dir_paths']):
        raise(Exception("model_dir_paths are not unique"))
    
    conf['n_models'] = len(conf['model_dir_paths'])
    conf['run_name'] = conf['run_name_format'].format(**conf)
    conf['save_dir_path'] = os.path.join(work_dir_path, f"ensemble/{conf['run_name']}")
    
    os.makedirs(conf['save_dir_path'], exist_ok=True)
    save_ensemble_conf_path = os.path.join(conf['save_dir_path'], f"{conf['run_name']}.json")
    utils.save_json(conf, save_ensemble_conf_path)

    return conf

def get_n_data(inferences, dataset_attribute):
    n_datas = [len(getattr(inference, dataset_attribute)) for inference in inferences]

    if len(set(n_datas)) != 1:
        raise(Exception("Number of images in dataset are not equal"))

    return n_datas[0]

def get_classes(inferences):
    classes = inferences[0].classes
    for i in range(1, len(inferences)):
        new_classes = inferences[i].classes
        if classes != new_classes:
            raise(Exception("Classes are not equal"))
    
    return classes


def ensemble_and_save(conf, mode):
    save_path = os.path.join(conf['save_dir_path'], f"{mode}_{conf['run_name']}.csv")

    if os.path.exists(save_path):
        print(f'{save_path} already exists')
        return

    inferences = []
    for model_dir_path in conf['model_dir_paths']:
        inference = Inference(model_dir_path=model_dir_path, TTA=conf['TTA'])
        inferences.append(inference)


    dataset_attribute = f'{mode}_dataset'
    n_data = get_n_data(inferences, dataset_attribute)
    classes = get_classes(inferences)

    filename_and_class = []
    rles = []

    for i in tqdm(range(n_data)): 
        outputs = None
        image_names = None

        for inference in inferences:
            if outputs is None:
                outputs, image_names = inference.inference(mode=mode, idx=i)
            else:
                new_outputs, new_image_names = inference.inference(mode=mode, idx=i)
                if new_image_names != image_names:
                    raise(Exception("Image names are different!"))
                outputs += new_outputs

        outputs = outputs > conf['n_models'] / 2

        if isinstance(image_names, str):
            image_names = (image_names, )

        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{classes['idx2class'][c]}_{image_name}")

    
    # 파일 저장하기
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(save_path, index=False)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ensemble_path', type=str, default=None) 
    args = parser.parse_args()
    return args



if __name__=='__main__':
    work_dir_path = os.path.dirname(os.path.realpath(__file__))

    args = parse_args()
    rel_ensemble_conf_path = args.ensemble_path
    conf = load_ensemble_conf(work_dir_path=work_dir_path, rel_ensemble_conf_path=rel_ensemble_conf_path)

    modes = ['test']
    for mode in modes:
        ensemble_and_save(conf, mode)