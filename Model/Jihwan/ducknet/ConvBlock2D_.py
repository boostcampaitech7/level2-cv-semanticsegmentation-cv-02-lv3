import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2D(nn.Module):
    def __init__(self, input_channel, filters, block_type, repeat=1, dilation_rate=1, size=3, device='cuda'):
        super(ConvBlock2D, self).__init__()
        self.block_type = block_type
        self.repeat = repeat
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.size = size
        self.device = device  # Device for the tensors

    def forward(self, x):
        x = x.to(self.device)  # Ensure input tensor is on the correct device
        result = x
        for _ in range(self.repeat):
            if self.block_type == 'separated':
                result = self.separated_conv2d_block(result, self.filters, self.size)
            elif self.block_type == 'duckv2':
                result = self.duckv2_conv2d_block(result, self.filters, self.size)
            elif self.block_type == 'midscope':
                result = self.midscope_conv2d_block(result, self.filters)
            elif self.block_type == 'widescope':
                result = self.widescope_conv2d_block(result, self.filters)
            elif self.block_type == 'resnet':
                result = self.resnet_conv2d_block(result, self.filters, self.dilation_rate)
            elif self.block_type == 'conv':
                padding = (self.size - 1) // 2
                conv = nn.Conv2d(result.size(1), self.filters, kernel_size=self.size, 
                               padding=padding)
                result = F.relu(conv(result))
            elif self.block_type == 'double_convolution':
                result = self.double_convolution_with_batch_norm(result, self.filters, self.dilation_rate)
            else:
                return None
        return result

    def duckv2_conv2d_block(self, x, filters, size):
        bn1 = nn.BatchNorm2d(x.size(1)).to(self.device)  # Ensure BN is on the correct device
        x = bn1(x)
        
        x1 = self.widescope_conv2d_block(x, filters)
        x2 = self.midscope_conv2d_block(x, filters)
        x3 = ConvBlock2D(filters, 'resnet', repeat=1, device=self.device)(x)
        x4 = ConvBlock2D(filters, 'resnet', repeat=2, device=self.device)(x)
        x5 = ConvBlock2D(filters, 'resnet', repeat=3, device=self.device)(x)
        x6 = self.separated_conv2d_block(x, filters, size=6)
        
        x = x1 + x2 + x3 + x4 + x5 + x6
        bn2 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        x = bn2(x)
        
        return x

    def separated_conv2d_block(self, x, filters, size=3):
        padding_h = (1 - 1) // 2  # For 1xsize kernel
        padding_w = (size - 1) // 2
        
        conv1 = nn.Conv2d(x.size(1), filters, kernel_size=(1, size), 
                        padding=(padding_h, padding_w)).to(self.device)
        bn1 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv2 = nn.Conv2d(filters, filters, kernel_size=(size, 1), 
                        padding=(padding_w, padding_h)).to(self.device)
        bn2 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        
        x = F.relu(conv1(x))
        x = bn1(x)
        x = F.relu(conv2(x))
        x = bn2(x)
        
        return x

    def midscope_conv2d_block(self, x, filters):
        padding1 = (3 - 1) // 2  # For 3x3 kernel
        padding2 = 2  # For dilation=2
        
        conv1 = nn.Conv2d(x.size(1), filters, kernel_size=3, padding=padding1, dilation=1).to(self.device)
        bn1 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=padding2, dilation=2).to(self.device)
        bn2 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        
        x = F.relu(conv1(x))
        x = bn1(x)
        x = F.relu(conv2(x))
        x = bn2(x)
        
        return x

    def widescope_conv2d_block(self, x, filters):
        padding1 = (3 - 1) // 2  # For dilation=1
        padding2 = 2  # For dilation=2
        padding3 = 3  # For dilation=3
        
        conv1 = nn.Conv2d(x.size(1), filters, kernel_size=3, padding=padding1, dilation=1).to(self.device)
        bn1 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=padding2, dilation=2).to(self.device)
        bn2 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=padding3, dilation=3).to(self.device)
        bn3 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        
        x = F.relu(conv1(x))
        x = bn1(x)
        x = F.relu(conv2(x))
        x = bn2(x)
        x = F.relu(conv3(x))
        x = bn3(x)
        
        return x

    def resnet_conv2d_block(self, x, filters, dilation_rate=1):
        padding = dilation_rate
        
        conv1 = nn.Conv2d(x.size(1), filters, kernel_size=1, padding=0, 
                        dilation=dilation_rate).to(self.device)
        conv2 = nn.Conv2d(x.size(1), filters, kernel_size=3, padding=padding, 
                        dilation=dilation_rate).to(self.device)
        bn1 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=padding, 
                        dilation=dilation_rate).to(self.device)
        bn2 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        bn3 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        
        x1 = F.relu(conv1(x))
        
        x = F.relu(conv2(x))
        x = bn1(x)
        x = F.relu(conv3(x))
        x = bn2(x)
        
        x = x + x1
        x = bn3(x)
        
        return x

    def double_convolution_with_batch_norm(self, x, filters, dilation_rate=1):
        padding = dilation_rate
        
        conv1 = nn.Conv2d(x.size(1), filters, kernel_size=3, padding=padding, 
                        dilation=dilation_rate).to(self.device)
        bn1 = nn.BatchNorm2d(filters).to(self.device)  # Ensure BN is on the correct device
        conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=padding, 
                        dilation=dilation_rate).to(self.device)
        bn2 = nn.BatchNorm2d(filters).to

        x = F.relu(conv1(x))
        x = bn1(x)
        x = F.relu(conv2(x))
        x = bn2(x)

        return x