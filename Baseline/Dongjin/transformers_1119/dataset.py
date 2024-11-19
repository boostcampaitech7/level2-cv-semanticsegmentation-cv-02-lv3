from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import utils


def get_xray_classes():
    classes = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
        
    class2idx = {v: i for i, v in enumerate(classes)}
    idx2class = {v: k for k, v in class2idx.items()}
    num_class = len(classes)

    dicts = {}
    dicts['classes'] = classes
    dicts['class2idx'] = class2idx
    dicts['idx2class'] = idx2class
    dicts['num_class'] = num_class

    return dicts


class XRayDataset(Dataset):
    def __init__(self, 
                 mode='train',
                 transforms=None,
                 image_processor=None,
                 data_dir_path=None,
                 data_info_path=None,
                 ):
        
        self.mode = mode
        self.transforms = transforms
        self.image_processor = image_processor
        self.data_dir_path = data_dir_path
        self.data_info_path = data_info_path

        self.image_paths, self.anns_paths = self.get_image_anns_paths() # image와 annotation 파일 경로 불러오기
        
        dicts = get_xray_classes() # xray 이미지 클래스 정보 불러오기
        self.num_class = dicts['num_class']
        self.class2idx = dicts['class2idx']

    
    def get_image_anns_paths(self):
        # data_info 파일에서 image와 annotation 상대경로 읽어서 절대경로로 변환
        # 예시 - test.json 파일
        # # test_filenames: [ID040/image1661319116107.png", "ID040/image1661319145363.png", ...]
        data_info = utils.read_json(self.data_info_path)
        image_paths = None
        anns_paths = None

        # train/valid dataset 이면
        if (self.mode == 'train') or (self.mode == 'valid'):
            base = 'train'
            rel_image_paths = data_info[f'{self.mode}_filenames']
            rel_anns_paths = data_info[f'{self.mode}_labelnames']

        # test datset 이면
        elif self.mode == 'test':
            base = 'test'
            rel_image_paths = data_info[f'{self.mode}_filenames']
            rel_anns_paths = None # annotation 정보가 없음

        # 상대경로를 절대경로로 변환
        image_paths = utils.combine_paths(f'{self.data_dir_path}/{base}/DCM', rel_image_paths)
        if rel_anns_paths is not None:
            anns_paths = utils.combine_paths(f'{self.data_dir_path}/{base}/outputs_json', rel_anns_paths)

        return image_paths, anns_paths
     

    def __getitem__(self, item):
        # 이미지 읽어오기
        image_path = self.image_paths[item]
        image = cv2.imread(image_path) 

        # annotation paths가 존재하면 
        # loss 계산 / evaluation에서 사용할 labels(mask) 생성
        if self.anns_paths is not None:
            anns_path = self.anns_paths[item]
            anns = utils.read_json(anns_path)['annotations']

            # (H, W, NC) 모양의 label을 생성
            label_shape = tuple(image.shape[:2]) + (self.num_class, )
            label = np.zeros(label_shape, dtype=np.uint8)

            # 클래스 별로 처리
            for ann in anns:
                class_name = ann["label"]
                class_ind = self.class2idx[class_name]
                points = np.array(ann["points"])

                # polygon 포맷을 dense mask 포맷으로 변환
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label

        # transforms가 있으면 image와 label(mask) 변환
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result['image']
            label = result['mask']

        # HxWxC -> CxHxW
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        result = self.image_processor(image, label) # 모델 입력에 맞는 image input으로 변경
        
        # 결과 반환 
        result['pixel_values'] = result['pixel_values'][0]
        result['labels'] =  np.stack(result['labels'])
        result['image'] = np.array(image)

        result['pixel_values'] = torch.from_numpy(result['pixel_values']).float()
        result['labels'] = torch.from_numpy(result['labels']).float()

        return result
    

    def __len__(self):
        return len(self.image_paths)
    

if __name__=='__main__':
    from transformers import AutoImageProcessor
    import matplotlib.pyplot as plt

    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # train.py의 디렉토리
    conf = utils.load_conf(work_dir_path=work_dir_path) 
    
    image_processor = AutoImageProcessor.from_pretrained(conf['model_name'], reduce_labels=True)
    train_dataset = XRayDataset(mode='train', 
                                transforms=None, 
                                image_processor=image_processor,
                                data_dir_path=conf['data_dir_path'],
                                data_info_path=conf['train_json_path'])
    