#!/usr/bin/python3
#coding=utf-8

###################
#try1_432.pth
##################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)

    def forward(self, left, down):
        self.sigmoid1 = nn.Softmax2d()
        self.sigmoid2 = nn.Softmax2d()

        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        avg_out1= torch.mean(out1v, dim=1, keepdim=True)
        max_out1, _ = torch.max(out1v, dim=1, keepdim=True)

        out1h = F.relu(self.bn2h(self.conv2h(left )), inplace=True)
        avg_out2 = torch.mean(out1h, dim=1, keepdim=True)
        max_out2, _ = torch.max(out1h, dim=1, keepdim=True)

        avg_out1_1 = avg_out1 * F.interpolate(max_out2, size=avg_out1.size()[2:], mode='bilinear')
        max_out1_1 = max_out1 * F.interpolate(avg_out2, size=max_out1.size()[2:], mode='bilinear')
        avg_out2_1 = avg_out2 * F.interpolate(max_out1, size=avg_out2.size()[2:], mode='bilinear')
        max_out2_1 = max_out2 * F.interpolate(avg_out1, size=max_out2.size()[2:], mode='bilinear')

        scale1 = torch.cat([avg_out1_1, max_out1_1], dim=1)
        scale1 = self.conv1(scale1)
        scale1 = F.interpolate(scale1, size=out1v.size()[2:], mode='bilinear')
        s = out1v * self.sigmoid1(scale1) + out1v

        scale = torch.cat([avg_out2_1, max_out2_1], dim=1)
        scale = self.conv2(scale)
        scale = F.interpolate(scale, size=out1h.size()[2:], mode='bilinear')
        out1h = out1h * self.sigmoid2(scale) + out1h
        out2v = F.relu(self.bn2v(self.conv2v(out1h)), inplace=True)
        if out2v.size()[2:] != s.size()[2:]:
            s = F.interpolate(s, size=out2v.size()[2:], mode='bilinear')

        fuse  = s*out2v
        fuse = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True)

        return fuse

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        d2 = F.interpolate(out3h, size=out2h.size()[2:], mode='bilinear')
        out2h1 = d2 * out2h

        d2 = F.interpolate(out4h, size=out3h.size()[2:], mode='bilinear')
        out3h1 = d2*out3h

        d2 = F.interpolate(out5v, size=out4h.size()[2:], mode='bilinear')
        out4h1 = d2 * out4h

        out4v = self.cfm45(out4h1 , out5v)
        out3v = self.cfm34(out3h1, out4v)
        pred = self.cfm23(out2h1, out3v)

        return out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)



class RFB(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),

        )
        self.branch21 = nn.Sequential(

            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch31 = nn.Sequential(

            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(2*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, in_channel, 3)

    def forward(self, x):
        self.conv_res(x)

        x2 = self.branch2(x)
        x = self.branch3(x)+x2

        x2 = self.branch21(x)
        x3 = self.branch31(x)

        x_cat = self.conv_cat(torch.cat((x2, x3), 1))

        x = self.relu(x_cat)
        return x

    def initialize(self):
        weight_init(self)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    def initialize(self):
        weight_init(self)

class F3Net(nn.Module):
    def __init__(self, cfg):
        super(F3Net, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.squeeze5 = RFB(2048, 64)
        self.squeeze4 = RFB(1024, 64)
        self.squeeze3 = RFB(512, 64)
        self.squeeze2 = RFB(256, 64)

        self.decoder1 = Decoder()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        out2h, out3h, out4h, out5v = self.bkbone(x)
        a, b, c, d = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        out3h, out4h, out5v, pred1 = self.decoder1(a, b, c, d)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        return pred1, out3h, out4h, out5h


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
