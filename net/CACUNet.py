"""
Network Definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import SimpleITK as sitk
dropout_rate = 0.2
num_aneurysmn = 2

INPUT_SIZE_1 = [48, 256, 256] 
INPUT_SIZE_2 = [48, 512, 512] 

#INPUT_SIZE_1 = [48, 160, 160]
#INPUT_SIZE_2 = [48, 160, 160] 

class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class AnisotropicMaxPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicMaxPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.MaxPool3d(kernel_size=(kernel_size[0], 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, kernel_size[1], 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 1, kernel_size[2]))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 3), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 3), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (3, 3, 1), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()
        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out


class AnisotropicAvgPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicAvgPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.AvgPool3d(kernel_size=(1, kernel_size[1], kernel_size[2]))
        self.pool4 = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, kernel_size[2]))
        self.pool5 = nn.AvgPool3d(kernel_size=(kernel_size[0], kernel_size[1], 1))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 1), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (1, 1, 3), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()
        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out

# Define a single 3D context-aware U-Net
class ResUNet(nn.Module):

    def __init__(self, training, inchannel, stage):
        """
        :param training: Define whether the network belongs to the training or testing phase
        :param inchannel: Number of input channels at the beginning of the network
        :param stage: Define whether the network belongs to the first or the second stage
        """
        super().__init__()

        self.training = training
        self.stage = stage

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, num_aneurysmn + 1, 1),
            nn.Softmax(dim=1)
        )
        
        self.map2 = nn.Sequential(
            nn.Conv3d(128, num_aneurysmn + 1, 1),
            nn.Softmax(dim=1)
        )
        
        self.map3 = nn.Sequential(
            nn.Conv3d(64, num_aneurysmn + 1, 1),
            nn.Softmax(dim=1)
        )
        
        self.map4 = nn.Sequential(
            nn.Conv3d(32, num_aneurysmn + 1, 1),
            nn.Softmax(dim=1)
        )
                   
        if stage == 'stage1':
            context_kernel_size1 = [i // 16 for i in INPUT_SIZE_1]
            self.context_block4 = AnisotropicMaxPooling(128, 128, kernel_size=context_kernel_size1, is_dynamic_empty_cache=True)
        else:
            context_kernel_size1 = [i // 16 for i in INPUT_SIZE_2]      
            self.context_block4 = AnisotropicMaxPooling(128, 128, kernel_size=context_kernel_size1, is_dynamic_empty_cache=True)

    def forward(self, inputs):

        if self.stage == 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)
              
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3) 

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)
        
        long_range4_context = self.context_block4(long_range4)#context_block

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4_context) + short_range4 
        outputs = F.dropout(outputs, dropout_rate, self.training)

        output1 = self.map1(outputs) 
                     
        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6 
        outputs = F.dropout(outputs, dropout_rate, self.training)

        output2 = self.map2(outputs) 
             
        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7 
        outputs = F.dropout(outputs, dropout_rate, self.training)
        
        output3 = self.map3(outputs)
        
        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs) 
                
        
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


# Defining context-aware cascaded U-Net (CACU-Net)
class Net(nn.Module):
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')
        self.stage2 = ResUNet(training=training, inchannel=num_aneurysmn + 2, stage='stage2')
        
    def forward(self, inputs):
        
        # The input data is first reduced by half in the axial direction and then fed into the first stage network
        inputs_stage1 = F.upsample(inputs, (48, 256, 256), mode='trilinear')

        # Obtaining results for the first stage at rough scales
        if self.training is True:
            output1_stage1, output2_stage1, output3_stage1, output4_stage1 = self.stage1(inputs_stage1)
            output1_stage1 = F.upsample(output1_stage1, (48, 512, 512), mode='trilinear') 
            output2_stage1 = F.upsample(output2_stage1, (48, 512, 512), mode='trilinear') 
            output3_stage1 = F.upsample(output3_stage1, (48, 512, 512), mode='trilinear') 
            output4_stage1 = F.upsample(output4_stage1, (48, 512, 512), mode='trilinear')          
            
        else:
            output4_stage1 = self.stage1(inputs_stage1)
            output4_stage1 = F.upsample(output4_stage1, (48, 512, 512), mode='trilinear') 
                         
        # The results of the first stage are spliced with the original input data and fed together into the second stage network
        inputs_stage2 = torch.cat((output4_stage1, inputs), dim=1)

        # Obtaining the results of the second stage
        if self.training is True:
            output1_stage2, output2_stage2, output3_stage2, output4_stage2 = self.stage2(inputs_stage2)
            output1_stage2 = F.upsample(output1_stage2, (48, 512, 512), mode='trilinear') 
            output2_stage2 = F.upsample(output2_stage2, (48, 512, 512), mode='trilinear') 
            output3_stage2 = F.upsample(output3_stage2, (48, 512, 512), mode='trilinear') 
        else:
            output4_stage2 = self.stage2(inputs_stage2)     
     
        if self.training is True:
            return output1_stage1, output2_stage1, output3_stage1, output4_stage1, output1_stage2, output2_stage2, output3_stage2, output4_stage2
        else:
            return output4_stage2


# Initialization of network parameters
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        try:
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)
            # print('kaiming_normal_initial')
        except BaseException as e:
            pass # print("Context")
        
net = Net(training=True)
net.apply(init)
