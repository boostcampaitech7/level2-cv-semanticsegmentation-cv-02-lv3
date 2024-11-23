from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import utils
import albumentations as A


def get_xray_classes(crop_type=None):
    classes = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

    finger_idx = list(range(0, 19))
    backhand_idx = list(range(19, 27))
    arm_idx = list(range(27, 29))
    fingerbackhand_idx = list(range(0, 27))
    
    idx = range(len(classes))

    if crop_type is None:
        pass
    elif crop_type == 'crop_finger':
        idx = finger_idx
    elif crop_type == 'crop_backhand':
        idx = backhand_idx
    elif crop_type == 'crop_arm':
        idx = arm_idx
    elif crop_type == 'crop_fingerbackhand':
        idx = fingerbackhand_idx
    else:
        raise(Exception(f"{crop_type} is not valid"))

    classes = np.array(classes)[idx].tolist()
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
                 crop_type=None,
                 crop_info=None,
                 transforms=None,
                 image_processor=None,
                 data_dir_path=None,
                 data_info_path=None,
                 debug=False
                 ):
        
        self.mode = mode
        self.crop_type = crop_type
        self.crop_info = crop_info
        self.transforms = transforms
        self.image_processor = image_processor
        self.data_dir_path = data_dir_path
        self.data_info_path = data_info_path
        self.debug = debug

        self.image_paths, self.anns_paths = self.get_image_anns_paths() # image와 annotation 파일 경로 불러오기
        if debug:
            self.image_paths = self.image_paths[0:16]
            self.anns_paths = self.anns_paths[0:16]
        
        dicts = get_xray_classes(crop_type) # xray 이미지 클래스 정보 불러오기
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
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path) 
        label = None
        result = None

        # annotation paths가 존재하면 
        # loss 계산 / evaluation에서 사용할 labels(mask) 생성
        if self.anns_paths is not None:
            label = self.load_label(item, image.shape)

        # crop 정보가 있으면 crop 수행
        crop_coodinate = None
        if self.crop_type is not None:
            x1, x2, y1, y2 = self.crop_info[image_name][self.crop_type]
            crop_coodinate = [x1, x2, y1, y2]
            transform = A.Compose([
                        A.Crop(x_min=x1, y_min=y1, x_max=x2+1, y_max=y2+1)])
            
            if label is not None:
                result = transform(image=image, mask=label)
                image = result['image']
                label = result['mask']
            else: 
                result = transform(image=image)
                image = result['image']

        # transforms가 있으면 image와 label(mask) 변환
        if self.transforms is not None:
            result = self.transforms(image=image, mask=label)
            image = result['image']
            label = result['mask']
            result['labels'] = label
            del result['mask']
            

        # image와 label 채널 순서 변경 (HxWxC -> CxHxW)
        image = image.transpose(2, 0, 1)
        if label is not None:
            label = label.transpose(2, 0, 1)

        if self.image_processor is not None: # image_processor가 정의되었으면
            if label is not None: # label이 정의되었으면
                result = self.image_processor(image, label) 
                result['labels'] = np.stack(result['labels'])
                result['labels'] = torch.from_numpy(result['labels']).float()
            else: # label이 정의되지 않았으면
                result = self.image_processor(image)

            # pixel_value torch로 저장
            result['pixel_values'] = result['pixel_values'][0]
            result['pixel_values'] = torch.from_numpy(result['pixel_values']).float()
        
        if result is None:
            result = {'image': image, 'labels': label}
        result['crop'] = crop_coodinate
            
        return result, image_name
    

    def load_label(self, item, image_shape):
        anns_path = self.anns_paths[item]
        anns = utils.read_json(anns_path)['annotations']

        # (H, W, NC) 모양의 label을 생성
        label_shape = tuple(image_shape[:2]) + (self.num_class, )
        label = np.zeros(label_shape, dtype=np.uint8)

        # 클래스 별로 처리
        for ann in anns:
            class_name = ann["label"]
            if class_name not in self.class2idx:
                continue

            class_ind = self.class2idx[class_name]
            points = np.array(ann["points"])

            # polygon 포맷을 dense mask 포맷으로 변환
            class_label = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        return label


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
    
