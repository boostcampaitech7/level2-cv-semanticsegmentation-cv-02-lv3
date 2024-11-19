from dataset import XRayDataset, get_xray_classes
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from torch.utils.data import DataLoader
from torch import nn, optim
import albumentations as A
from tqdm import tqdm

import utils
import torch
import os
import wandb


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
        self.load_model() # model, optimizer, loss function 준비


    def train(self):
        best_valid_dice = 0

        for epoch in range(1, self.conf['max_epoch']+1):
            # train 실행 
            train_log = self.train_valid(epoch, mode='train', thr=0.5)
            
            # valid 실행
            with torch.no_grad():
                valid_log = self.train_valid(epoch, mode='valid', thr=0.5)

            
            if best_valid_dice < valid_log['dice']:
                best_valid_dice = valid_log['dice']
                save_path = os.path.join(self.conf['model_dir_path'], f'ep_{epoch}_dice_{best_valid_dice:.4f}')
                self.save_model(save_path)
                                         
                print(f'New best_valid_dice is achieved: {best_valid_dice:.4f} - {save_path}')                        

            # 로그 출력 및 기록
            self.log(epoch, train_log, valid_log) 


    def save_model(self, save_dir_path):
        os.makedirs(save_dir_path, exist_ok=True)
        self.model.save_pretrained(save_dir_path)
        self.image_processor.save_pretrained(save_dir_path)

    def train_valid(self, epoch, mode='train', thr=0.5):
        losses = []
        dices = []

        print(f'{epoch} - {mode} start')

        if mode == 'train':
            self.model.train()
            loader = self.train_loader
        elif mode == 'valid':
            self.model.eval()
            loader = self.valid_loader
        else:
            raise(Exception(f"{mode} is not valid"))

        with tqdm(loader, unit="batch") as tepoch:
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

        # 손실 평균
        avg_loss = utils.average(losses)
        
        # dice 계산
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        avg_dice = torch.mean(dices_per_class).item()

        dicts = {c: v.item() for c, v in zip(self.xray_classes['classes'], dices_per_class)}

        results = {}
        results['loss'] = avg_loss
        results['dice'] = avg_dice
        results['dices_per_class'] = dicts

        return results
     
    

 
    def log(self, epoch, train_log, valid_log):
        log_path = os.path.join(self.conf['model_dir_path'], 'log.json')

        if os.path.exists(log_path):
            logs = utils.read_json(log_path)
        else:
            logs = {'epoch': {}}
        
        dicts = {}
        for k, v in train_log.items():
            dicts['train_'+k] = v
        for k, v in valid_log.items():
            dicts['valid_'+k] = v

        logs['epoch'][epoch] = dicts
        print(f"epoch: {epoch} - train_loss: {dicts['train_loss']:4f}, train_dice: {dicts['train_loss']:4f}, valid_loss: {dicts['valid_loss']:4f}, valid_dice: {dicts['valid_dice']:4f}")
        
        wandb.log(dicts)
        utils.save_json(logs, log_path)

        



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
                                    data_info_path=self.conf['train_json_path'],
                                    debug=self.conf['debug'])
        
        self.valid_dataset = XRayDataset(mode='valid', 
                                    transforms=None, 
                                    image_processor=self.image_processor,
                                    data_dir_path=self.conf['data_dir_path'],
                                    data_info_path=self.conf['valid_json_path'],
                                    debug=self.conf['debug'])
        
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