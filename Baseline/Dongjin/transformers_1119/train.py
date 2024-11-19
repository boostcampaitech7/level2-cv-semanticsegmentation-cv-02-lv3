from dataset import XRayDataset, get_xray_classes
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from torch.utils.data import DataLoader
from torch import nn, optim
import albumentations as A
from tqdm import tqdm

import utils
import torch


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.load_dataloader() # train_loader, valid_loader 준비
        self.load_model() # model, optimzier, loss function 준비

    def train_valid(self, mode='train', thr=0.5):
        losses = []
        dices = []
        
        if mode == 'train':
            self.model.train()
        elif mode == 'valid':
            self.model.eval()
        else:
            raise(Exception(f"{mode} is not valid"))

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                if mode == 'train': # train이면 graident 초기화
                    self.optimizer.zero_grad()

                inputs = batch["pixel_values"].to(self.conf['device'])
                labels = batch["labels"].to(self.conf['device'])
                outputs = self.model(inputs).logits

                output_h, output_w = outputs.size(-2), outputs.size(-1)
                label_h, label_w = labels.size(-2), labels.size(-1)

                # label과 output size가 다르면 upsampling
                if output_h != label_h or output_w != label_w:
                    outputs = nn.functional.interpolate(outputs, size=(label_h, label_w), mode="bilinear")

                loss = self.loss_func(outputs, labels)
                loss_item = loss.item()
                losses.append(loss_item)
                
                if mode == 'train': # train이면 gradient 계산 및 업데이트
                    loss.backward()
                    self.optimizer.step()

                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr)

                dice = dice_coef(outputs, labels)
                dices.append(dice.detach().cpu())
                tepoch.set_postfix(loss=loss_item)

        avg_loss = utils.average(losses)
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)

        dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(self.xray_classes['classes'], dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        print(dice_str)

        avg_dice = torch.mean(dices_per_class).item()

        return avg_loss, avg_dice, dices_per_class



    def train(self):
        for epoch in range(self.conf['max_epoch']):
            losses = []
            print(f'epoch: {epoch+1} - train')
            train_loss, train_dice, train_dices_per_class = self.train_valid(mode='train', thr=0.5)
            print(f'epoch: {epoch+1}, train loss: {train_loss:.4f}, train dice: {train_dice:.4f}')

            print(f'epoch: {epoch+1} - valid')
            with torch.no_grad():
                valid_loss, valid_dice, valid_dices_per_class = self.train_valid(mode='valid', thr=0.5)
            print(f'epoch: {epoch+1}, valid loss: {valid_loss:.4f}, valid dice: {valid_dice:.4f}')


    def load_train_transforms(self):
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
        
        if self.conf['debug'] == True:
            self.train_dataset = self.train_dataset[0:self.conf['batch_size'] * 2]
            self.valid_dataset = self.valid_dataset[0:self.conf['batch_size'] * 2]

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf['batch_size'], shuffle=True, num_workers=self.conf['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.conf['batch_size'], shuffle=False, num_workers=self.conf['num_workers'])
    

    def load_model(self):
        self.xray_classes = get_xray_classes()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.conf['model_name'],
                    num_labels=self.xray_classes['num_class'],
                    id2label=self.xray_classes['idx2class'],
                    label2id=self.xray_classes['class2idx']).to(self.conf['device'])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.conf['learning_rate'])
        self.loss_func = nn.BCEWithLogitsLoss()