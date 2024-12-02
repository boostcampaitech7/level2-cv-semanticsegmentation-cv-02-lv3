#swin-t + unet3+
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.models.swin_transformer import swin_tiny_patch4_window7_224  # Swin-Tiny 모델
from torchvision.models import swin_v2_t
from torchvision.models.swin_transformer import Swin_V2_T_Weights


class UNet3Plus_Swin(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4):
        super(UNet3Plus_Swin, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        # Swin Transformer encoder
        swin_weights = Swin_V2_T_Weights.DEFAULT
        swin = swin_v2_t(weights=swin_weights)
        self.stage1 = swin.features[0]  # Stage 1 (Patch Embedding)
        self.stage2 = swin.features[1]  # Stage 2
        self.stage3 = swin.features[2]  # Stage 3
        self.stage4 = swin.features[3]  # Stage 4
        self.norm = swin.features[4]    # LayerNorm for final output

        # Feature map sizes from Swin Transformer
        filters = [96, 192, 384, 768, 1024]  # Swin V2 Tiny feature sizes
        self.CatChannels = filters[0]  # 96
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks  #96*5하면 480채널이 되긴 함.

        #############Decoder####################################################
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1) # (96 -> 96)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # ConvTranspose2d 추가
        #self.h1_upsample_hd4 = nn.ConvTranspose2d(       # 480 -> 480
        #    in_channels=self.CatChannels,
        #    out_channels=self.CatChannels,
        #    kernel_size=2,
        #    stride=2
        #)
        #이미지 사이즈 63*63
        # ConvTranspose2d 정의: 각 Swin Transformer 출력 레이어의 크기 조정
        #self.h3_upsample = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=2, stride=2)  # 63x63 -> 126x126

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1) #96 -> 96
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)


        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1) #192 -> 96
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1) #192-> 96
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1) #384 -> 96
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 480 -> 480
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)


        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)  #192 -> 96
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)  #192->96
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)  
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)  ###**
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)


        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)
        ####################################################


        # ConvTranspose2d 정의: 각 Swin Transformer 출력 레이어의 크기 조정
        self.h3_upsample = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=2, stride=2)  # 63x63 -> 126x126
        self.h4_upsample = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=2, stride=2)  # 63x63 -> 126x126
        self.h5_upsample = nn.ConvTranspose2d(in_channels=384, out_channels=96, kernel_size=4, stride=4)  # 32x32 -> 126x126

        # Final output layer
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, kernel_size=3, padding=1)


    def forward(self, x):
        # Encoder
        h1 = self.stage1(x)  # First stage (patch embedding)
        h2 = self.stage2(h1)  # Second stage
        h3 = self.stage3(h2)  # Third stage
        h4 = self.stage4(h3)  # Fourth stage
        h5 = self.norm(h4)    # Final normalization layer

        # Convert channels_last (B, H, W, C) -> channels_first (B, C, H, W)
        h1 = h1.permute(0, 3, 1, 2)
        h2 = h2.permute(0, 3, 1, 2)
        h3 = h3.permute(0, 3, 1, 2)
        h4 = h4.permute(0, 3, 1, 2)
        h5 = h5.permute(0, 3, 1, 2)


        #print("h1.shape : " , h1.shape)
        #print("h2.shape : " , h2.shape)
        #print("h3.shape : " , h3.shape)
        #print("h4.shape : " , h4.shape)
        #print("h5.shape : " , h5.shape) #h5.shape :  torch.Size([4, 2048, 16, 16])

        # Ensure consistent sizes (이걸 왜 하지)
        #hd5 = F.interpolate(h5, size=(h5.size(2), h5.size(3)), mode='bilinear', align_corners=False)
        #print("hd5.shape : " , hd5.shape)  #torch.Size([4, 64, 16, 16])
        # Decoder
        # Combine resized encoder outputs
        ## -------------Decoder------------
        target_size = (154, 154)

        ###############h4############################
        #h1_PT_hd4 :  torch.Size([2, 96, 16, 16])
        #h2_PT_hd4 :  torch.Size([2, 96, 32, 32])
        #h3_PT_hd4 :  torch.Size([2, 96, 32, 32])
        #h4_Cat_hd4 :  torch.Size([2, 96, 63, 63])
        #hd5_UT_hd4 :  torch.Size([2, 96, 64, 64])

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        #print("h1_PT_hd4 : ", h1_PT_hd4.shape)
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        #print("h2_PT_hd4 : ", h2_PT_hd4.shape)
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        #print("h3_PT_hd4 : ", h3_PT_hd4.shape)
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        #print("h4_Cat_hd4 : ", h4_Cat_hd4.shape)
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(h5))))
        #print("hd5_UT_hd4 : ", hd5_UT_hd4.shape)

        #####서로다른 이미지 사이즈 126으로 동일하게 맞추기
        h1_PT_hd4 = F.interpolate(h1_PT_hd4, size=target_size, mode='bilinear', align_corners=False)
        h2_PT_hd4 = F.interpolate(h2_PT_hd4, size=target_size, mode='bilinear', align_corners=False)
        h3_PT_hd4 = F.interpolate(h3_PT_hd4, size=target_size, mode='bilinear', align_corners=False)
        h4_Cat_hd4 = F.interpolate(h4_Cat_hd4, size=target_size, mode='bilinear', align_corners=False)
        hd5_UT_hd4 = F.interpolate(hd5_UT_hd4, size=target_size, mode='bilinear', align_corners=False)

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), dim=1)))) # hd4->40*40*UpChannels
        #print("hd4 : ", hd4.shape)


        ###############h3##############################
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        # ConvTranspose2d 업샘플링 적용
        #h1_PT_hd3 = self.h1_upsample_hd3(h1_PT_hd3)
        #print("h1_PT_hd3 : ", h1_PT_hd3.shape)
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        #print("h2_PT_hd3 : ", h2_PT_hd3.shape)
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        #print("h3_Cat_hd3 : ", h3_Cat_hd3.shape)
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        #print("hd4_UT_hd3 : ", hd4_UT_hd3.shape)
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(h5))))
        #print("hd5_UT_hd3 : ", hd5_UT_hd3.shape)

        h1_PT_hd3 = F.interpolate(h1_PT_hd3, size=target_size, mode='bilinear', align_corners=False)
        h2_PT_hd3 = F.interpolate(h2_PT_hd3, size=target_size, mode='bilinear', align_corners=False)
        h3_Cat_hd3 = F.interpolate(h3_Cat_hd3, size=target_size, mode='bilinear', align_corners=False)
        hd4_UT_hd3 = F.interpolate(hd4_UT_hd3, size=target_size, mode='bilinear', align_corners=False)
        hd5_UT_hd3 = F.interpolate(hd5_UT_hd3, size=target_size, mode='bilinear', align_corners=False)

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels
        #print("hd3 : ", hd3.shape)


        ###############h2############################
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        # ConvTranspose2d 업샘플링 적용
        #h1_PT_hd2 = self.h1_upsample_hd2(h1_PT_hd2)
        #print("h1_PT_hd2 : ", h1_PT_hd2.shape)
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        #print("h2_Cat_hd2 : ", h2_Cat_hd2.shape)
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        #print("hd3_UT_hd2 : ", hd3_UT_hd2.shape)
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        #print("hd4_UT_hd2 : ", hd4_UT_hd2.shape)
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(h5))))

        h1_PT_hd2 = F.interpolate(h1_PT_hd2, size=target_size, mode='bilinear', align_corners=False)
        h2_Cat_hd2 = F.interpolate(h2_Cat_hd2, size=target_size, mode='bilinear', align_corners=False)
        hd3_UT_hd2 = F.interpolate(hd3_UT_hd2, size=target_size, mode='bilinear', align_corners=False)
        hd4_UT_hd2 = F.interpolate(hd4_UT_hd2, size=target_size, mode='bilinear', align_corners=False)
        hd5_UT_hd2 = F.interpolate(hd5_UT_hd2, size=target_size, mode='bilinear', align_corners=False)


        #print("hd5_UT_hd2 : ", hd5_UT_hd2.shape)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels
        #print("hd2 : ", hd2.shape)


        ###############h1############################
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        #print("h1_Cat_hd1 : ", h1_Cat_hd1.shape)
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        #print("hd2_UT_hd1 : ", hd2_UT_hd1.shape)
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        #print("hd3_UT_hd1 : ", hd3_UT_hd1.shape)
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        #print("hd4_UT_hd1 : ", hd4_UT_hd1.shape)
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(h5))))
        #print("hd5_UT_hd1 : ", hd5_UT_hd1.shape)

        h1_Cat_hd1 = F.interpolate(h1_Cat_hd1, size=target_size, mode='bilinear', align_corners=False)
        hd2_UT_hd1 = F.interpolate(hd2_UT_hd1, size=target_size, mode='bilinear', align_corners=False)
        hd3_UT_hd1 = F.interpolate(hd3_UT_hd1, size=target_size, mode='bilinear', align_corners=False)
        hd4_UT_hd1 = F.interpolate(hd4_UT_hd1, size=target_size, mode='bilinear', align_corners=False)
        hd5_UT_hd1 = F.interpolate(hd5_UT_hd1, size=target_size, mode='bilinear', align_corners=False)

        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        #print("hd1 : ", hd1.shape)

        # **여기에서 입력 크기를 기반으로 hd1 크기를 맞춤**
        # x.size()에서 입력 크기를 가져옴
        output_size = (x.size(2), x.size(3))  # 입력 이미지의 높이와 너비
        #전체 concat한거 이미지 올리기
        hd1 = F.interpolate(hd1, size=output_size, mode='bilinear', align_corners=False) #616

        #print(h1_PT_hd4.shape, h2_PT_hd4.shape, h3_PT_hd4.shape, h4_Cat_hd4.shape, hd5_UT_hd4.shape)

        d1 = self.outconv1(hd1)
        #print("d1.shape : ", d1.shape)
        return d1
#print("---------------------------------------------------------------------------------------------")