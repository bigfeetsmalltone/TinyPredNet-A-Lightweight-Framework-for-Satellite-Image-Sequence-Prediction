import torch
import torch.nn as nn
from torch.nn import functional as F

import copy

import torch
from torch import nn
from modules import ConvSC
import math

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class AGMSTB(nn.Module):

    def __init__(self, ratio, inplanes, planes, stride=1, reduced_dim=32, scale = 8, expansion = 8):
        super(AGMSTB, self).__init__()

        self.expansion = expansion

        width = int(math.floor(planes * (reduced_dim/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width

        self.last_pooling = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Sequential(
            nn.Linear(width*scale, width*scale // ratio),
            nn.ReLU(),
            nn.Linear(width*scale // ratio, width*scale),
            nn.Sigmoid())

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)

        b, c, _, _ = out.size()
        y = self.last_pooling(out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class TinyPredNet(nn.Module):
    def __init__(self, shape_in=[8, 1, 256, 256], ratio = 16, in_channels=512, out_channels=64, reduced_dim = 32, scale=8, expansion=8, blocks=4, hid_S=16, N_S=4):
        super(TinyPredNet, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        # self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        # self.hid = AGMSTB(in_channels, out_channels, scale = scale, expansion = expansion)
        hids = []
        for i in range(blocks):
            hids.append(AGMSTB(ratio, in_channels, out_channels, reduced_dim = reduced_dim, scale = scale, expansion = expansion))
        self.hid = nn.Sequential(*hids)
        
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw): 
        B, T, C, H, W = x_raw.shape
        x = x_raw.contiguous().view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        # z = embed.contiguous().view(B, T, C_, H_, W_)
        z = embed.contiguous().view(B, T*C_, H_, W_)
        
        hid = self.hid(z)
        
        hid = hid.reshape(B*T, C_, H_, W_)


        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y



class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_length = args.out_len
        self.input_length = args.short_len
        self.model = TinyPredNet(in_channels = args.in_channels, out_channels = args.out_channels, reduced_dim = args.reduced_dim, scale = args.scale, expansion = args.expansion, blocks = args.blocks,  hid_S = args.hid_S, N_S = args.N_S)

    def forward(self, batch_x):
        cnt = self.output_length // self.input_length
        pred_y = []
        temp = batch_x
        # print(temp.shape)
        for i in range(cnt):
            temp = self.model(temp)
            pred_y.append(temp)
        pred_y = torch.cat(pred_y, 1)
        return pred_y
