from dataset import XRayDataset, get_xray_classes
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from torch.utils.data import DataLoader
from torch import nn, optim
import albumentations as A
from tqdm import tqdm

import utils



def upsampling(input, output_size):
    output = nn.functional.interpolate(
            input,
            size=output_size,
            mode="bilinear")
    
    return output



class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.load_dataloader() # train_loader, valid_loader 준비
        self.load_model_and_optimizer() # model과 optimizer 준비

    def train(self):
        for epoch in range(self.conf['max_epoch']):
            losses = []
            
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.optimizer.zero_grad()
                    inputs = batch["pixel_values"].to(self.conf['device'])
                    labels = batch["labels"].to(self.conf['device'])
                    outputs = self.mode(inputs).logits

                    labels_size = labels.shape[-2:]
                    outputs_resize = upsampling(outputs, labels_size)

                    loss = self.loss_func(outputs_resize, labels)
                    loss_item = loss.item()
                    losses.append(loss_item)

                    loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(loss_item)

            avg_loss = utils.average(losses)


    def load_train_transforms():
        train_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
            ])
        return train_transforms


    def load_dataloader(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.conf['model_name'], reduce_labels=True)
        self.train_dataset = XRayDataset(mode='train', 
                                    transforms=self.load_train_transforms(), 
                                    image_processor=self.image_processor,
                                    data_dir_path=self.conf['data_dir_path'],
                                    data_info_path=self.conf['train_json_path'])
        
        self.valid_dataset = XRayDataset(mode='valid', 
                                    transforms=None, 
                                    image_processor=self.image_processor,
                                    data_dir_path=self.conf['data_dir_path'],
                                    data_info_path=self.conf['valid_json_path'])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf['batch_size'], shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.conf['batch_size'], shuffle=False)
    

    def load_model(self):
        self.xray_classes = get_xray_classes()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.conf['model_name'],
                    num_labels=self.xray_classes['num_class'],
                    id2label=self.xray_classes['idx2class'],
                    label2id=self.xray_classes['class2idx']).to(self.conf['device'])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.conf['learning_rate'])
        self.loss_func = nn.BCEWithLogitsLoss()