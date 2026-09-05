# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class BackboneHead(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(BackboneHead, self).__init__()
        self.module_0 = py_nndct.nn.Input() #BackboneHead::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[0]/Conv2d[conv]/input.3
        self.module_2 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[0]/LeakyReLU[act]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[1]/Conv2d[conv]/input.9
        self.module_4 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[1]/LeakyReLU[act]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Conv[cv1]/Conv2d[conv]/input.15
        self.module_6 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Conv[cv1]/LeakyReLU[act]/9179
        self.module_7 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/9184
        self.module_8 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/9204
        self.module_9 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.21
        self.module_10 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.25
        self.module_11 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.27
        self.module_12 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/9273
        self.module_13 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Bottleneck[m]/ModuleList[0]/9275
        self.module_14 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/input.31
        self.module_15 = py_nndct.nn.Conv2d(in_channels=48, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Conv[cv2]/Conv2d[conv]/input.33
        self.module_16 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[2]/Conv[cv2]/LeakyReLU[act]/input.37
        self.module_17 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[3]/Conv2d[conv]/input.39
        self.module_18 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[3]/LeakyReLU[act]/input.43
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Conv[cv1]/Conv2d[conv]/input.45
        self.module_20 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Conv[cv1]/LeakyReLU[act]/9359
        self.module_21 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/9364
        self.module_22 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/9384
        self.module_23 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.51
        self.module_24 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.55
        self.module_25 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.57
        self.module_26 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/9453
        self.module_27 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[0]/input.61
        self.module_28 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[1]/Conv[cv1]/Conv2d[conv]/input.63
        self.module_29 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[1]/Conv[cv1]/LeakyReLU[act]/input.67
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[1]/Conv[cv2]/Conv2d[conv]/input.69
        self.module_31 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[1]/Conv[cv2]/LeakyReLU[act]/9509
        self.module_32 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Bottleneck[m]/ModuleList[1]/9511
        self.module_33 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/input.73
        self.module_34 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Conv[cv2]/Conv2d[conv]/input.75
        self.module_35 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[4]/Conv[cv2]/LeakyReLU[act]/input.79
        self.module_36 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[5]/Conv2d[conv]/input.81
        self.module_37 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[5]/LeakyReLU[act]/input.85
        self.module_38 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Conv[cv1]/Conv2d[conv]/input.87
        self.module_39 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Conv[cv1]/LeakyReLU[act]/9595
        self.module_40 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/9600
        self.module_41 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/9620
        self.module_42 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.93
        self.module_43 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.97
        self.module_44 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.99
        self.module_45 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/9689
        self.module_46 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[0]/input.103
        self.module_47 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[1]/Conv[cv1]/Conv2d[conv]/input.105
        self.module_48 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[1]/Conv[cv1]/LeakyReLU[act]/input.109
        self.module_49 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[1]/Conv[cv2]/Conv2d[conv]/input.111
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[1]/Conv[cv2]/LeakyReLU[act]/9745
        self.module_51 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Bottleneck[m]/ModuleList[1]/9747
        self.module_52 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/input.115
        self.module_53 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Conv[cv2]/Conv2d[conv]/input.117
        self.module_54 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[6]/Conv[cv2]/LeakyReLU[act]/input.121
        self.module_55 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[7]/Conv2d[conv]/input.123
        self.module_56 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[7]/LeakyReLU[act]/input.127
        self.module_57 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Conv[cv1]/Conv2d[conv]/input.129
        self.module_58 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Conv[cv1]/LeakyReLU[act]/9831
        self.module_59 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/9836
        self.module_60 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/9856
        self.module_61 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.135
        self.module_62 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.139
        self.module_63 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.141
        self.module_64 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/9925
        self.module_65 = py_nndct.nn.Add() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Bottleneck[m]/ModuleList[0]/9927
        self.module_66 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/input.145
        self.module_67 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Conv[cv2]/Conv2d[conv]/input.147
        self.module_68 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[8]/Conv[cv2]/LeakyReLU[act]/input.151
        self.module_69 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/Conv[cv1]/Conv2d[conv]/input.153
        self.module_70 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/MaxPool2d[m]/9997
        self.module_71 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/MaxPool2d[m]/10011
        self.module_72 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/MaxPool2d[m]/10025
        self.module_73 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/input.155
        self.module_74 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/Conv[cv2]/Conv2d[conv]/input.157
        self.module_75 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/SPPF[model]/Sequential[model]/SPPF[9]/Conv[cv2]/LeakyReLU[act]/input.161
        self.module_76 = py_nndct.nn.Interpolate() #BackboneHead::BackboneHead/Upsample[model]/Sequential[model]/Upsample[10]/10060
        self.module_77 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/Concat[model]/Sequential[model]/Concat[11]/input.163
        self.module_78 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Conv[cv1]/Conv2d[conv]/input.165
        self.module_79 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Conv[cv1]/LeakyReLU[act]/10090
        self.module_80 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/10095
        self.module_81 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/10115
        self.module_82 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.171
        self.module_83 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.175
        self.module_84 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.177
        self.module_85 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/10184
        self.module_86 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/input.181
        self.module_87 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Conv[cv2]/Conv2d[conv]/input.183
        self.module_88 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[12]/Conv[cv2]/LeakyReLU[act]/input.187
        self.module_89 = py_nndct.nn.Interpolate() #BackboneHead::BackboneHead/Upsample[model]/Sequential[model]/Upsample[13]/10219
        self.module_90 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/Concat[model]/Sequential[model]/Concat[14]/input.189
        self.module_91 = py_nndct.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Conv[cv1]/Conv2d[conv]/input.191
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Conv[cv1]/LeakyReLU[act]/10249
        self.module_93 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/10254
        self.module_94 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/10274
        self.module_95 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.197
        self.module_96 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.201
        self.module_97 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.203
        self.module_98 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/10343
        self.module_99 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/input.207
        self.module_100 = py_nndct.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Conv[cv2]/Conv2d[conv]/input.209
        self.module_101 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[15]/Conv[cv2]/LeakyReLU[act]/input.213
        self.module_102 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[16]/Conv2d[conv]/input.215
        self.module_103 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[16]/LeakyReLU[act]/10400
        self.module_104 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/Concat[model]/Sequential[model]/Concat[17]/input.219
        self.module_105 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Conv[cv1]/Conv2d[conv]/input.221
        self.module_106 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Conv[cv1]/LeakyReLU[act]/10430
        self.module_107 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/10435
        self.module_108 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/10455
        self.module_109 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.227
        self.module_110 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.231
        self.module_111 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.233
        self.module_112 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/10524
        self.module_113 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/input.237
        self.module_114 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Conv[cv2]/Conv2d[conv]/input.239
        self.module_115 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[18]/Conv[cv2]/LeakyReLU[act]/input.243
        self.module_116 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[19]/Conv2d[conv]/input.245
        self.module_117 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Conv[model]/Sequential[model]/Conv[19]/LeakyReLU[act]/10581
        self.module_118 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/Concat[model]/Sequential[model]/Concat[20]/input.249
        self.module_119 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Conv[cv1]/Conv2d[conv]/input.251
        self.module_120 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Conv[cv1]/LeakyReLU[act]/10611
        self.module_121 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/10616
        self.module_122 = py_nndct.nn.strided_slice() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/10636
        self.module_123 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.257
        self.module_124 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/LeakyReLU[act]/input.261
        self.module_125 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.263
        self.module_126 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/LeakyReLU[act]/10705
        self.module_127 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/input.267
        self.module_128 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Conv[cv2]/Conv2d[conv]/input.269
        self.module_129 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/C2f[model]/Sequential[model]/C2f[21]/Conv[cv2]/LeakyReLU[act]/input.321
        self.module_130 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[0]/Conv[0]/Conv2d[conv]/input.273
        self.module_131 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[0]/Conv[0]/LeakyReLU[act]/input.277
        self.module_132 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[0]/Conv[1]/Conv2d[conv]/input.279
        self.module_133 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[0]/Conv[1]/LeakyReLU[act]/input.283
        self.module_134 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[0]/Conv2d[2]/10808
        self.module_135 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[0]/Conv[0]/Conv2d[conv]/input.285
        self.module_136 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[0]/Conv[0]/LeakyReLU[act]/input.289
        self.module_137 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[0]/Conv[1]/Conv2d[conv]/input.291
        self.module_138 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[0]/Conv[1]/LeakyReLU[act]/input.295
        self.module_139 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[0]/Conv2d[2]/10881
        self.module_140 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/10884
        self.module_141 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[1]/Conv[0]/Conv2d[conv]/input.297
        self.module_142 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[1]/Conv[0]/LeakyReLU[act]/input.301
        self.module_143 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[1]/Conv[1]/Conv2d[conv]/input.303
        self.module_144 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[1]/Conv[1]/LeakyReLU[act]/input.307
        self.module_145 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[1]/Conv2d[2]/10957
        self.module_146 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[1]/Conv[0]/Conv2d[conv]/input.309
        self.module_147 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[1]/Conv[0]/LeakyReLU[act]/input.313
        self.module_148 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[1]/Conv[1]/Conv2d[conv]/input.315
        self.module_149 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[1]/Conv[1]/LeakyReLU[act]/input.319
        self.module_150 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[1]/Conv2d[2]/11030
        self.module_151 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/11033
        self.module_152 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[2]/Conv[0]/Conv2d[conv]/input.323
        self.module_153 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[2]/Conv[0]/LeakyReLU[act]/input.327
        self.module_154 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[2]/Conv[1]/Conv2d[conv]/input.329
        self.module_155 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[2]/Conv[1]/LeakyReLU[act]/input.333
        self.module_156 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv2]/ModuleList[2]/Conv2d[2]/11106
        self.module_157 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[2]/Conv[0]/Conv2d[conv]/input.335
        self.module_158 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[2]/Conv[0]/LeakyReLU[act]/input.339
        self.module_159 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[2]/Conv[1]/Conv2d[conv]/input.341
        self.module_160 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[2]/Conv[1]/LeakyReLU[act]/input
        self.module_161 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #BackboneHead::BackboneHead/Sequential[model]/Sequential[model]/Detect[22]/ModuleList[cv3]/ModuleList[2]/Conv2d[2]/11179
        self.module_162 = py_nndct.nn.Cat() #BackboneHead::BackboneHead/11182

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_7 = self.module_7(input=output_module_0, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,16,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_8 = self.module_8(input=output_module_0, dim=[0,1,2,3], start=[0,16,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_9 = self.module_9(output_module_8)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_9 = self.module_12(output_module_9)
        output_module_13 = self.module_13(input=output_module_8, other=output_module_9, alpha=1)
        output_module_7 = self.module_14(dim=1, tensors=[output_module_7,output_module_8,output_module_13])
        output_module_7 = self.module_15(output_module_7)
        output_module_7 = self.module_16(output_module_7)
        output_module_7 = self.module_17(output_module_7)
        output_module_7 = self.module_18(output_module_7)
        output_module_7 = self.module_19(output_module_7)
        output_module_7 = self.module_20(output_module_7)
        output_module_21 = self.module_21(input=output_module_7, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,32,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_22 = self.module_22(input=output_module_7, dim=[0,1,2,3], start=[0,32,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_23 = self.module_23(output_module_22)
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_23 = self.module_26(output_module_23)
        output_module_27 = self.module_27(input=output_module_22, other=output_module_23, alpha=1)
        output_module_28 = self.module_28(output_module_27)
        output_module_28 = self.module_29(output_module_28)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(output_module_28)
        output_module_32 = self.module_32(input=output_module_27, other=output_module_28, alpha=1)
        output_module_21 = self.module_33(dim=1, tensors=[output_module_21,output_module_22,output_module_27,output_module_32])
        output_module_21 = self.module_34(output_module_21)
        output_module_21 = self.module_35(output_module_21)
        output_module_36 = self.module_36(output_module_21)
        output_module_36 = self.module_37(output_module_36)
        output_module_36 = self.module_38(output_module_36)
        output_module_36 = self.module_39(output_module_36)
        output_module_40 = self.module_40(input=output_module_36, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,64,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_41 = self.module_41(input=output_module_36, dim=[0,1,2,3], start=[0,64,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_42 = self.module_42(output_module_41)
        output_module_42 = self.module_43(output_module_42)
        output_module_42 = self.module_44(output_module_42)
        output_module_42 = self.module_45(output_module_42)
        output_module_46 = self.module_46(input=output_module_41, other=output_module_42, alpha=1)
        output_module_47 = self.module_47(output_module_46)
        output_module_47 = self.module_48(output_module_47)
        output_module_47 = self.module_49(output_module_47)
        output_module_47 = self.module_50(output_module_47)
        output_module_51 = self.module_51(input=output_module_46, other=output_module_47, alpha=1)
        output_module_40 = self.module_52(dim=1, tensors=[output_module_40,output_module_41,output_module_46,output_module_51])
        output_module_40 = self.module_53(output_module_40)
        output_module_40 = self.module_54(output_module_40)
        output_module_55 = self.module_55(output_module_40)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_59 = self.module_59(input=output_module_55, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,128,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_60 = self.module_60(input=output_module_55, dim=[0,1,2,3], start=[0,128,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_61 = self.module_61(output_module_60)
        output_module_61 = self.module_62(output_module_61)
        output_module_61 = self.module_63(output_module_61)
        output_module_61 = self.module_64(output_module_61)
        output_module_65 = self.module_65(input=output_module_60, other=output_module_61, alpha=1)
        output_module_59 = self.module_66(dim=1, tensors=[output_module_59,output_module_60,output_module_65])
        output_module_59 = self.module_67(output_module_59)
        output_module_59 = self.module_68(output_module_59)
        output_module_59 = self.module_69(output_module_59)
        output_module_70 = self.module_70(output_module_59)
        output_module_71 = self.module_71(output_module_70)
        output_module_72 = self.module_72(output_module_71)
        output_module_73 = self.module_73(dim=1, tensors=[output_module_59,output_module_70,output_module_71,output_module_72])
        output_module_73 = self.module_74(output_module_73)
        output_module_73 = self.module_75(output_module_73)
        output_module_76 = self.module_76(input=output_module_73, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_76 = self.module_77(dim=1, tensors=[output_module_76,output_module_40])
        output_module_76 = self.module_78(output_module_76)
        output_module_76 = self.module_79(output_module_76)
        output_module_80 = self.module_80(input=output_module_76, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,64,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_81 = self.module_81(input=output_module_76, dim=[0,1,2,3], start=[0,64,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_82 = self.module_82(output_module_81)
        output_module_82 = self.module_83(output_module_82)
        output_module_82 = self.module_84(output_module_82)
        output_module_82 = self.module_85(output_module_82)
        output_module_80 = self.module_86(dim=1, tensors=[output_module_80,output_module_81,output_module_82])
        output_module_80 = self.module_87(output_module_80)
        output_module_80 = self.module_88(output_module_80)
        output_module_89 = self.module_89(input=output_module_80, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_89 = self.module_90(dim=1, tensors=[output_module_89,output_module_21])
        output_module_89 = self.module_91(output_module_89)
        output_module_89 = self.module_92(output_module_89)
        output_module_93 = self.module_93(input=output_module_89, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,32,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_94 = self.module_94(input=output_module_89, dim=[0,1,2,3], start=[0,32,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_95 = self.module_95(output_module_94)
        output_module_95 = self.module_96(output_module_95)
        output_module_95 = self.module_97(output_module_95)
        output_module_95 = self.module_98(output_module_95)
        output_module_93 = self.module_99(dim=1, tensors=[output_module_93,output_module_94,output_module_95])
        output_module_93 = self.module_100(output_module_93)
        output_module_93 = self.module_101(output_module_93)
        output_module_102 = self.module_102(output_module_93)
        output_module_102 = self.module_103(output_module_102)
        output_module_102 = self.module_104(dim=1, tensors=[output_module_102,output_module_80])
        output_module_102 = self.module_105(output_module_102)
        output_module_102 = self.module_106(output_module_102)
        output_module_107 = self.module_107(input=output_module_102, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,64,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_108 = self.module_108(input=output_module_102, dim=[0,1,2,3], start=[0,64,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_109 = self.module_109(output_module_108)
        output_module_109 = self.module_110(output_module_109)
        output_module_109 = self.module_111(output_module_109)
        output_module_109 = self.module_112(output_module_109)
        output_module_107 = self.module_113(dim=1, tensors=[output_module_107,output_module_108,output_module_109])
        output_module_107 = self.module_114(output_module_107)
        output_module_107 = self.module_115(output_module_107)
        output_module_116 = self.module_116(output_module_107)
        output_module_116 = self.module_117(output_module_116)
        output_module_116 = self.module_118(dim=1, tensors=[output_module_116,output_module_73])
        output_module_116 = self.module_119(output_module_116)
        output_module_116 = self.module_120(output_module_116)
        output_module_121 = self.module_121(input=output_module_116, dim=[0,1,2,3], start=[0,0,0,0], end=[9223372036854775807,128,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_122 = self.module_122(input=output_module_116, dim=[0,1,2,3], start=[0,128,0,0], end=[9223372036854775807,9223372036854775807,9223372036854775807,9223372036854775807], step=[1,1,1,1])
        output_module_123 = self.module_123(output_module_122)
        output_module_123 = self.module_124(output_module_123)
        output_module_123 = self.module_125(output_module_123)
        output_module_123 = self.module_126(output_module_123)
        output_module_121 = self.module_127(dim=1, tensors=[output_module_121,output_module_122,output_module_123])
        output_module_121 = self.module_128(output_module_121)
        output_module_121 = self.module_129(output_module_121)
        output_module_130 = self.module_130(output_module_93)
        output_module_130 = self.module_131(output_module_130)
        output_module_130 = self.module_132(output_module_130)
        output_module_130 = self.module_133(output_module_130)
        output_module_130 = self.module_134(output_module_130)
        output_module_135 = self.module_135(output_module_93)
        output_module_135 = self.module_136(output_module_135)
        output_module_135 = self.module_137(output_module_135)
        output_module_135 = self.module_138(output_module_135)
        output_module_135 = self.module_139(output_module_135)
        output_module_130 = self.module_140(dim=1, tensors=[output_module_130,output_module_135])
        output_module_141 = self.module_141(output_module_107)
        output_module_141 = self.module_142(output_module_141)
        output_module_141 = self.module_143(output_module_141)
        output_module_141 = self.module_144(output_module_141)
        output_module_141 = self.module_145(output_module_141)
        output_module_146 = self.module_146(output_module_107)
        output_module_146 = self.module_147(output_module_146)
        output_module_146 = self.module_148(output_module_146)
        output_module_146 = self.module_149(output_module_146)
        output_module_146 = self.module_150(output_module_146)
        output_module_141 = self.module_151(dim=1, tensors=[output_module_141,output_module_146])
        output_module_152 = self.module_152(output_module_121)
        output_module_152 = self.module_153(output_module_152)
        output_module_152 = self.module_154(output_module_152)
        output_module_152 = self.module_155(output_module_152)
        output_module_152 = self.module_156(output_module_152)
        output_module_157 = self.module_157(output_module_121)
        output_module_157 = self.module_158(output_module_157)
        output_module_157 = self.module_159(output_module_157)
        output_module_157 = self.module_160(output_module_157)
        output_module_157 = self.module_161(output_module_157)
        output_module_152 = self.module_162(dim=1, tensors=[output_module_152,output_module_157])
        return (output_module_130,output_module_141,output_module_152)
