import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock2D(nn.Module):
    def __init__(self, filters, block_type, repeat=1, dilation_rate=1, size=3, padding=1):
        super(ConvBlock2D, self).__init__()
        self.filters = filters
        self.block_type = block_type
        self.repeat = repeat
        self.dilation_rate = dilation_rate
        self.size = size
        self.padding = padding
    def forward(self, x):
        result = x
        for _ in range(self.repeat):
            if self.block_type == 'separated':
                result = SeparatedConv2DBlock(self.filters, self.size, self.padding)(result)
            elif self.block_type == 'duckv2':
                result = DuckV2Conv2DBlock(self.filters, self.size)(result)
            elif self.block_type == 'midscope':
                result = MidScopeConv2DBlock(self.filters)(result)
            elif self.block_type == 'widescope':
                result = WideScopeConv2DBlock(self.filters)(result)
            elif self.block_type == 'resnet':
                result = ResNetConv2DBlock(self.filters, self.dilation_rate)(result)
            elif self.block_type == 'conv':
                result = nn.ReLU()(nn.Conv2d(x.shape[1], self.filters, kernel_size=self.size, padding=self.padding)(result))
            elif self.block_type == 'double_convolution':
                result = DoubleConvolutionWithBatchNormalization(self.filters, self.dilation_rate)(result)
            else:
                return None
        return result
class DuckV2Conv2DBlock(nn.Module):
    def __init__(self, filters, size):
        super(DuckV2Conv2DBlock, self).__init__()
        self.filters = filters
        self.size = size
    def forward(self, x):
        x1 = WideScopeConv2DBlock(self.filters)(x)
        x2 = MidScopeConv2DBlock(self.filters)(x)
        x3 = ResNetConv2DBlock(self.filters)(x)
        x4 = ResNetConv2DBlock(self.filters)(x)
        x5 = ResNetConv2DBlock(self.filters)(x)
        x6 = SeparatedConv2DBlock(self.filters, size=self.size)(x)
        print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        x = x1 + x2 + x3 + x4 + x5 + x6
        return nn.BatchNorm2d(x.shape[1])(x)
class SeparatedConv2DBlock(nn.Module):
    def __init__(self, filters, size=3, padding=1):
        super(SeparatedConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(1, size), padding=padding)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=padding)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x
class MidScopeConv2DBlock(nn.Module):
    def __init__(self, filters):
        super(MidScopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=2)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x
class WideScopeConv2DBlock(nn.Module):
    def __init__(self, filters):
        super(WideScopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, filters, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=2)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=3)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = F.relu(self.bn(self.conv3(x)))
        return x
class ResNetConv2DBlock(nn.Module):
    def __init__(self, filters, dilation_rate=1):
        super(ResNetConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=1, padding=0, dilation=dilation_rate)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.bn(self.conv2(x)))
        x_final = F.relu(x1 + x2)
        return x_final
class DoubleConvolutionWithBatchNormalization(nn.Module):
    def __init__(self, filters, dilation_rate=1):
        super(DoubleConvolutionWithBatchNormalization, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x