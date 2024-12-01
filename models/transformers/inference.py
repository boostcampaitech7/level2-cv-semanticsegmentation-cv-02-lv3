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


# mask map을 RLE로 변환
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


# RLE를 mask map으로 변환 
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

        self.load_dataset() # dataloader 불러오기 
        self.classes = get_xray_classes(self.conf['crop_type']) # xray 사진의 class 지정


    def load_dataset(self):
        # crop_info 불러오기
        crop_info = None
        if self.conf['crop_info_path']:
            crop_info = utils.read_json(self.conf['crop_info_path'])

        # train_dataset 불러오기 
        self.train_dataset = XRayDataset(mode='train',
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info, 
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['train_json_path'])

        # valid_dataset 불러오기 
        self.valid_dataset = XRayDataset(mode='valid', 
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info,
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['valid_json_path'])

        # test_dataset 불러오기 
        self.test_dataset = XRayDataset(mode='test', 
                                        crop_type=self.conf['crop_type'],
                                        crop_info=crop_info,
                                        transforms=None, 
                                        image_processor=self.image_processor,
                                        data_dir_path=self.conf['data_dir_path'],
                                        data_info_path=self.conf['test_json_path'])


    def hflip_crop(self, crop):
        # 이미지에 hflip을 수행했을 때 변경된 crop 좌표 구하기
        w, h = [2048, 2048]
        x1, x2, y1, y2 = crop
        x2, x1 = w-x1-1, w-x2-1
        return x1, x2, y1, y2


    def predict(self, inputs, crop):
        # 입력 이미지에 대한 예측 수행
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
            
        inputs = inputs.to('cuda') # 입력 이미지
        outputs = self.model(inputs).logits # 모델 출력
        
        if crop is None:
            # 입력 이미지에 crop을 수행하지 않았으면, 원본 이미지 사이즈 (2048, 2048)로 interpolate 수행
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        else: 
            # 입력 이미지에 crop을 수행했으면
            # 모델 출력 결과를 crop 크기로 interpolate 수행
            # 원본 이미지 사이즈 크기로 초기화된 배열(temp)의 crop 좌표에 interpolate 결과 입력
            INF = 100
            x1, x2, y1, y2 = crop
            w, h = x2-x1+1, y2-y1+1 
            temp = F.interpolate(outputs, size=(h, w), mode="bilinear")
            outputs = -INF * torch.ones(temp.shape[0], temp.shape[1], 2048, 2048).to('cuda')
            outputs[:, :, y1:y2+1, x1:x2+1] = temp
            del temp

        # 원본 이미지 사이즈와 동일한 모델 예측결과(+sigmoid) 반환
        return outputs.sigmoid()


    def inference(self, mode, idx):
        # 지정된 mode와 idx에 대한 추론 수행
        if mode == 'train':
            dataset = self.train_dataset
        elif mode == 'valid':
            dataset = self.valid_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise(Exception("mode supports train/valid/test"))

        # dataset으로부터 입력이미지(inputs)와 crop 정보 받아오기
        batch, image_names = dataset[idx]
        inputs = batch["pixel_values"]
        crop = batch['crop']

        if self.TTA: # TTA 수행 (원본 + horizontal flip 예측 결과) 
            outputs = self.predict(inputs, crop)
            outputs += self.predict(inputs.flip(dims=[-1]), self.hflip_crop(crop)).flip(dims=[-1])
            outputs = outputs / 2
        else: # TTA 수행 X
            outputs = self.predict(inputs, crop)

        # 결과 반환
        return outputs.detach().cpu(), image_names


def load_ensemble_conf(work_dir_path, rel_ensemble_conf_path):
    # ensemble_conf 파일 불러오기
    ensemble_conf_path = os.path.join(work_dir_path, rel_ensemble_conf_path)
    conf = utils.read_json(ensemble_conf_path)
    conf['model_dir_paths'] = []

    # model_dir_path_format를 이용하여 model_dir_paths 지정
    for model_dir_path_format in conf['model_dir_path_formats']:
        model_dir_path = model_dir_path_format.format(**conf)
        conf['model_dir_paths'].append(model_dir_path)

    # model_dir_paths에 동일한 경로가 있으면 오류 반환
    if not utils.is_unique(conf['model_dir_paths']):
        raise(Exception("model_dir_paths are not unique"))
    
    conf['n_models'] = len(conf['model_dir_paths'])
    conf['run_name'] = conf['run_name_format'].format(**conf)
    conf['save_dir_path'] = os.path.join(work_dir_path, f"ensemble/{conf['run_name']}")
    
    os.makedirs(conf['save_dir_path'], exist_ok=True) # 폴더 생성
    save_ensemble_conf_path = os.path.join(conf['save_dir_path'], f"{conf['run_name']}.json")
    utils.save_json(conf, save_ensemble_conf_path) # 예측에 사용할 conf를 저장

    return conf

def get_n_data(inferences, dataset_attribute):
    # inference 별 dataset_attribute에 해당하는 이미지 갯수가 동일한지 확인
    # 동일하면 이미지 갯수 반환
    n_datas = [len(getattr(inference, dataset_attribute)) for inference in inferences]

    if len(set(n_datas)) != 1:
        raise(Exception("Number of images in dataset are not equal"))

    return n_datas[0]

def get_classes(inferences):
    # inference 별 예측하는 class가 동일한지 확인 및 class 반환
    classes = inferences[0].classes
    for i in range(1, len(inferences)):
        new_classes = inferences[i].classes
        if classes != new_classes:
            raise(Exception("Classes are not equal"))
    
    return classes


def ensemble_and_save(conf, mode):
    # 앙상블 예측(단일모델도 가능) 및 저장 수행
    save_path = os.path.join(conf['save_dir_path'], f"{mode}_{conf['run_name']}.csv") # 결과 저장 경로 설정

    # 결과 저장 경로에 파일이 존재하면 종료
    if os.path.exists(save_path):
        print(f'{save_path} already exists')
        return

    # 모델들 불러오기
    inferences = []
    for model_dir_path in conf['model_dir_paths']:
        inference = Inference(model_dir_path=model_dir_path, TTA=conf['TTA'])
        inferences.append(inference)

    # 예측해야 할 이미지 수 및 클래스 불러오기
    dataset_attribute = f'{mode}_dataset'
    n_data = get_n_data(inferences, dataset_attribute)
    classes = get_classes(inferences)

    filename_and_class = []
    rles = []

    for i in tqdm(range(n_data)): 
        outputs = None
        image_names = None

        for inference in inferences: # 각 모델 불러오기
            if outputs is None: # 첫 번째로 수행되는 예측
                outputs, image_names = inference.inference(mode=mode, idx=i)
            else: # n 번째로 수행되는 예측
                new_outputs, new_image_names = inference.inference(mode=mode, idx=i)
                if new_image_names != image_names: # 모델 별 결과를 예측한 이미지가 모두 동일한지 확인
                    raise(Exception("Image names are different!"))
                outputs += new_outputs # 예측 결과 더하기

        # 모델 출력 평균이 > 1/2보다 크면 예측 마스크 1로 변환 
        outputs = outputs > conf['n_models'] / 2

        # image_names가 str이면 tuple 타입으로 변환 
        if isinstance(image_names, str):
            image_names = (image_names, )

        # 마스크를 rle로 변환
        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{classes['idx2class'][c]}_{image_name}")

    
    # 파일 저장을 위한 전처리
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    # 파일 저장
    df.to_csv(save_path, index=False)


# argument parser 기능 구현
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ensemble_path', type=str, default=None) 
    args = parser.parse_args()
    return args



if __name__=='__main__':
    # inference.py는 단일 및 다수 모델의 앙상블 예측 결과를 얻을 수 있습니다. 
    # json 파일을 이용하여 어떤 단일 혹은 다수 모델의 결과 예측을 수행할지 지정 가능합니다. 
    # 사용 예시: python inference.py --ensemble_path conf/ensemble.json

    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # 기본 경로 설정 

    # argument 기능 구현
    args = parse_args()
    rel_ensemble_conf_path = args.ensemble_path
    conf = load_ensemble_conf(work_dir_path=work_dir_path, rel_ensemble_conf_path=rel_ensemble_conf_path)


    modes = ['train', 'valid', 'test']
    for mode in modes:
        ensemble_and_save(conf, mode) # 앙상블 예측 및 저장 수행 (단일 모델도 가능)