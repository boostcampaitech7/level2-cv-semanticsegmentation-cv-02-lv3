import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, input_channel, filters, block_type, repeat=1, dilation_rate=1, size=3, padding=1):
        super(ConvBlock2D, self).__init__()
        self.filters = filters
        self.block_type = block_type
        self.repeat = repeat
        self.dilation_rate = dilation_rate
        self.size = size
        self.padding = padding
        self.SeparatedConv2DBlock = SeparatedConv2DBlock(input_channel, filters, size, padding)
        self.DuckV2Conv2DBlock = DuckV2Conv2DBlock(input_channel, filters, size)
        self.MidScopeConv2DBlock = MidScopeConv2DBlock(input_channel,filters)
        self.WideScopeConv2DBlock = WideScopeConv2DBlock(input_channel, filters)
        self.ResNetConv2DBlock = ResNetConv2DBlock(input_channel, filters, dilation_rate)
        # self.DoubleConvolutionWithBatchNormalization = DoubleConvolutionWithBatchNormalization(input_channel, filters, dilation_rate)
        self.activation = nn.ReLU()
        self.basic_conv = nn.Conv2d(3, self.filters, kernel_size=self.size, padding=self.padding)

    def forward(self, x):
        result = x
        for _ in range(self.repeat):
            if self.block_type == 'separated':
                result = self.SeparatedConv2DBlock(result)
            elif self.block_type == 'duckv2':
                result = self.DuckV2Conv2DBlock(result)
            elif self.block_type == 'midscope':
                result = self.MidScopeConv2DBlock(result)
            elif self.block_type == 'widescope':
                result = self.WideScopeConv2DBlock(result)  
            elif self.block_type == 'resnet':
                result = self.ResNetConv2DBlock(result)
            elif self.block_type == 'double_convolution':
                result = self.DoubleConvolutionWithBatchNormalization(result)
            else:
                return None
        return result

    
class DuckV2Conv2DBlock(nn.Module):
    def __init__(self, input_channel, filters, size):
        super(DuckV2Conv2DBlock, self).__init__()
        self.filters = filters
        self.size = size

        # 미리 레이어 초기화
        self.x1_block = WideScopeConv2DBlock(input_channel, filters)
        self.x2_block = MidScopeConv2DBlock(input_channel, filters)
        self.x3_block = ResNetConv2DBlock(input_channel, filters)
        self.x4_blocks = nn.ModuleList([
            ResNetConv2DBlock(input_channel if i == 0 else filters, filters)
            for i in range(2)
        ])
        self.x5_blocks = nn.ModuleList([
            ResNetConv2DBlock(input_channel if i == 0 else filters, filters)
            for i in range(3)
        ])
        self.x6_block = SeparatedConv2DBlock(input_channel, filters, size=size)
        
        self.batch_norm = nn.BatchNorm2d(self.filters)


    def forward(self, x):
        x1 = self.x1_block(x)
        x2 = self.x2_block(x)
        x3 = self.x3_block(x)
        x4 = x
        for block in self.x4_blocks:
            x4 = block(x4)
        x5 = x
        for block in self.x5_blocks:
            x5 = block(x5)
        x6 = self.x6_block(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        return self.batch_norm(x)

class SeparatedConv2DBlock(nn.Module):
    def __init__(self, input_channel, filters, size=3, padding=1):
        super(SeparatedConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, filters, kernel_size=(1, size), padding=(0, size // 2))
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=(size // 2, 0))
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x


class MidScopeConv2DBlock(nn.Module):
    def __init__(self, input_channel, filters):
        super(MidScopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, filters, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x



class WideScopeConv2DBlock(nn.Module):
    def __init__(self, input_channel, filters):
        super(WideScopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, filters, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=3)
        self.bn = nn.BatchNorm2d(filters)
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = F.relu(self.bn(self.conv3(x)))
        return x


class ResNetConv2DBlock(nn.Module):
    def __init__(self, input_channel, filters, dilation_rate=1):
        super(ResNetConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, filters, kernel_size=1, padding=0, dilation=dilation_rate)
        self.conv2 = nn.Conv2d(input_channel, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.bn3 = nn.BatchNorm2d(filters)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.bn1(self.conv2(x)))
        x3 = F.relu(self.bn2(self.conv3(x2)))
        x_final = F.relu(x1 + x2)
        x_final = self.bn3(x_final)
        return x_final


# class DoubleConvolutionWithBatchNormalization(nn.Module):
#     def __init__(self, filters, dilation_rate=1):
#         super(DoubleConvolutionWithBatchNormalization, self).__init__()
#         self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.bn = nn.BatchNorm2d(filters)
#     def forward(self, x):
#         x = F.relu(self.bn(self.conv1(x)))
#         x = F.relu(self.bn(self.conv2(x)))
#         return x
    



class DUCKNet(nn.Module):
    def __init__(self, img_height: int, img_width: int, input_channels: int, 
                 out_classes: int, starting_filters: int):
        super(DUCKNet, self).__init__()
        self.input_channels = input_channels
        self.starting_filters = starting_filters
        
        # Initial downsampling convolutions
        self.down_conv1 = nn.Conv2d(input_channels, starting_filters * 2, 
                                   kernel_size=2, stride=2, padding=0)
        self.down_conv2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, 
                                   kernel_size=2, stride=2, padding=0)
        self.down_conv3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, 
                                   kernel_size=2, stride=2, padding=0)
        self.down_conv4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, 
                                   kernel_size=2, stride=2, padding=0)
        self.down_conv5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, 
                                   kernel_size=2, stride=2, padding=0)

        # Initial path convolutions
        self.conv_l1i = nn.Conv2d(starting_filters, starting_filters * 2, 
                                 kernel_size=2, stride=2, padding=0)
        self.conv_l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, 
                                 kernel_size=2, stride=2, padding=0)
        self.conv_l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, 
                                 kernel_size=2, stride=2, padding=0)
        self.conv_l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, 
                                 kernel_size=2, stride=2, padding=0)
        self.conv_l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, 
                                 kernel_size=2, stride=2, padding=0)

        # ConvBlock2D layers (using the previously defined ConvBlock2D class)
        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        self.t1 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=1)
        self.t55 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'resnet', repeat=1)

        # Decoder ConvBlock2D layers
        self.q4 = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)
        self.q3 = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)
        self.q6 = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)
        self.q1 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        
        # Final output layer
        self.final_conv = nn.Conv2d(starting_filters, out_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling path
        p1 = self.down_conv1(F.pad(x, (0, 1, 0, 1)))  # Padding to match 'same'
        p2 = self.down_conv2(F.pad(p1, (0, 1, 0, 1)))
        p3 = self.down_conv3(F.pad(p2, (0, 1, 0, 1)))
        p4 = self.down_conv4(F.pad(p3, (0, 1, 0, 1)))
        p5 = self.down_conv5(F.pad(p4, (0, 1, 0, 1)))

        # Initial processing
        t0 = self.t0(x)
        
        # Encoder path
        l1i = self.conv_l1i(F.pad(t0, (0, 1, 0, 1)))
        s1 = l1i + p1
        t1 = self.t1(s1)
        
        l2i = self.conv_l2i(F.pad(t1, (0, 1, 0, 1)))
        s2 = l2i + p2
        t2 = self.t2(s2)
        
        l3i = self.conv_l3i(F.pad(t2, (0, 1, 0, 1)))
        s3 = l3i + p3
        t3 = self.t3(s3)
        
        l4i = self.conv_l4i(F.pad(t3, (0, 1, 0, 1)))
        s4 = l4i + p4
        t4 = self.t4(s4)
        
        l5i = self.conv_l5i(F.pad(t4, (0, 1, 0, 1)))
        s5 = l5i + p5

        
        # Bridge
        t51 = self.t51(s5)
        t53 = self.t53(t51)
        t55 = self.t55(t53)
        # Decoder path with skip connections
        l5o = F.interpolate(t55, scale_factor=2, mode='nearest')
        c4 = l5o + t4

        q4 = self.q4(c4)
        
        l4o = F.interpolate(q4, scale_factor=2, mode='nearest')
        c3 = l4o + t3
        q3 = self.q3(c3)
        
        l3o = F.interpolate(q3, scale_factor=2, mode='nearest')
        c2 = l3o + t2
        q6 = self.q6(c2)
        
        l2o = F.interpolate(q6, scale_factor=2, mode='nearest')
        c1 = l2o + t1
        q1 = self.q1(c1)
        
        l1o = F.interpolate(q1, scale_factor=2, mode='nearest')
        c0 = l1o + t0
        z1 = self.z1(c0)
        
        # Final output
        output = self.final_conv(z1)
        return output