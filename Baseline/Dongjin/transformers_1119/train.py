from dataset import XRayDataset
from transformers import AutoImageProcessor

class Trainer:
    def __init__(self, conf):
        self.conf = conf

    def get_dataloader(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.conf['model_name'], reduce_labels=True)
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
