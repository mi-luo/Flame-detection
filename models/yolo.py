# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""
#from mmcv.cnn import kaiming_init
import argparse
import sys
from copy import deepcopy
from pathlib import Path
# ============================自己设计的解耦检测头=========================
from models.common import *
from collections import OrderedDict
from functools import partial
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from typing import Tuple
from torch import Tensor
from models.GML import  GML_mask, Hard, Easy
# ============可能用到的模块=====================
# ================通道解耦注意力======================
import torch.nn as nn
import torch
# ===== S2-MLPv2 注意力=========
# S2-MLPv2 注意力！
# for yolov5 ing!!!!
# 改进 定位任务需要进行空间移位后的通道注意力，分类任务只需要通道注意力
import numpy as np
import torch
from torch import nn
from torch.nn import init
from .MPN import MPNCOV
class PSA_t(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_t, self).__init__()
        # print("C S attention")
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)
        self.covar_pool = MPNCOV(self.inplanes)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_init(self.conv_q_right, mode='fan_in')
        # kaiming_init(self.conv_v_right, mode='fan_in')
        # kaiming_init(self.conv_q_left, mode='fan_in')
        # kaiming_init(self.conv_v_left, mode='fan_in')
        #
        # self.conv_q_right.inited = True
        # self.conv_v_right.inited = True
        # self.conv_q_left.inited = True
        # self.conv_v_left.inited = True
        pass

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)  # 1x1卷积

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)  # 1x1卷积

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask) # H*W维度上进行softmax

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)  # 1x1 SE

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x) # 1x1卷积

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width) # 1x1卷积

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out


    def covariance_pool(self, x):
        y = self.covar_pool(x)
        y = y.unsqueeze(dim=1)
        y = self.avg_pool(y)

        return x * y

    def forward(self, x1, x2 ,x3):
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3, 1, 2)
        x3 = x3.permute(0,3, 1, 2)
        # [N, C, H, W]
        x1_context_channel = self.channel_pool(x1)

        # [N, C, H, W]

        # context_channel = self.channel_pool(x2)
        # print(x2.shape, x2.dtype)
        # x2 = x2.to(torch.float32)
        x2_context_covariance = self.covariance_pool(x2)
        x3_context_spatial = self.spatial_pool(x3)
        # context_covariance = context_covariance.to(x1.dtype)
        # [N, C, H, W]
        return x1_context_channel.permute(0, 2,3,1), x2_context_covariance.permute(0, 2,3,1) ,x3_context_spatial.permute(0, 2,3,1)# 返回值ontext_covariance怎么安排？？

# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


class Res(nn.Module):
    def __init__(self, d, channels):
        super(Res, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(d, d, 1, 1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            nn.Conv2d(d, d, 1, 1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            nn.Conv2d(d, channels, 1, 1),
            nn.GroupNorm(8, channels),
        )
        self.m2 = nn.Conv2d(d, channels, 1, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu( self.m1(x) + self.m2(x))


class S2Attention(nn.Module):

    def __init__(self, channels=256): # channels=256
        #     def __init__(self, channels=3 ):
        super().__init__()
        d = channels  # 2-> 128 4-> 64
        self.se = nn.Conv2d(channels, d, 1, 1)
        # self.exp1 = nn.Conv2d(d, channels, 1, 1)
        # self.exp2 = nn.Conv2d(d, channels, 1, 1)
        # print("change exp")
        self.exp1 = nn.Sequential(
            Res(d, d),
            Res(d, d),
            Res(d, channels),
        )  # 64 80, 128, 40, 256 20
        self.exp2 = nn.Sequential(
            Res(d, d),
            Res(d, d),
            Res(d, channels),
        )
        self.mlp1 = nn.Linear(d, d * 3)
        # self.mlp2 = nn.Linear(d, d)  # 第一个任务 回归 self.split_attention则输入通道为d
        # self.mlp3 = nn.Linear(d, d)  # 第二个任务 分类 self.split_attention则输入通道为d
        # print("----",d) # d=128
        self.mlp2 = nn.Linear(2*d, d)  # 第一个任务 回归
        self.mlp3 = nn.Linear(2*d, d)  # 第二个任务 分类
        # self.split_attention = SplitAttention(d)
        self.psa = PSA_t(d, d)


    def forward(self, x):
        # print("--------",x.shape)
        x = self.se(x)  # 通道为d
        b, c, w, h = x.size() # c=d
        x = x.permute(0, 2, 3, 1)
        # print("-------x",x.shape)
        x = self.mlp1(x)  # 3c
        # print("++++++x",x.shape)
        # x1 = spatial_shift1(x[:, :, :, :c])  # 转移   c  ???
        x1 = x[:, :, :, :c]
        x2 = x[:, :, :, c:c * 2]  # 非转移  c
        x3 = x[:, :, :, c * 2:]  # 非转移  c
        x1, x2, x3 = self.psa(x1, x2, x3)
        #         x_all=torch.stack([x1,x2,x3],1)
        # =================自己的通道融合方式===========
        # x_all_1 = torch.stack([x1, x2], -1) # 通道
        # x_all_2 = torch.stack([x2, x3], -1) # 回归
        x_all_1 = torch.cat([x1, x2], -1) # 通道
        # print(x_all_1.shape)
        x_all_2 = torch.cat([x2, x3], -1) # 回归

        x1 = self.mlp2(x_all_1)  # 分类
        x2 = self.mlp3(x_all_2)  # 回归
        # print(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        return x1, x2

# ================通道解耦注意力======================

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class DecoupledHead(nn.Module):
    # 代码参考： https://blog.csdn.net/weixin_44119362
    def __init__(self, ch=256, nc=80, width=1.0, anchors=(), ind = 0):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3

        #===============================
        # if ind == 0:
        #     d = 256 # 大特征图 感觉可以用256？？？
        # if ind == 1:
        #     d = 256
        # if ind == 2:
        #     d = 256 # 可能也需要改？？
        # ===============================

        self.merge = Conv(ch, 256 * width, 1, 1)  # BCHW  80 #
        # print("self.merge",self.merge)
        # self.cls_att = SE(256*width, ratio=2)
        self.S2 = S2Attention(256)  # S2Attention的channels=256  # 添加注意力来看解耦效果！！！！！！！！！！！！！！！！
        # self.S2 = S2Attention(d)  # S2Attention的channels=256  # 添加注意力来看解耦效果！！！！！！！！！！！！！！！！
        self.cls_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        # self.cls_convs1 = double(256 * width, 256 * width, )
        self.cls_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        # self.cls_convs2 = double(256 * width, 256 * width, )
        # self.reg_att = SE(256*width, ratio=2)
        self.reg_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)
        # self.BiLevelRoutingAttention = BiLevelRoutingAttention(256)
        # self.BiLevelRoutingAttention = BiLevelRoutingAttention(256)



    def forward(self, x):
        # print("DecoupledHead in ",x.shape)
        """
        DecoupledHead in  torch.Size([1, 128, 32, 32])
        DecoupledHead in  torch.Size([1, 256, 16, 16])
        DecoupledHead in  torch.Size([1, 512, 8, 8])
        """
        x = self.merge(x)  # 这里的x不是Nan
        # print("----",x.shape) # torch.Size([1, 64, 32, 32])
        # print("----------",x.shape)
        # print("DecoupledHead in 经过merge后的",x)
        #     # 分类=3x3conv + 3x3conv + 1x1convpred
        x1, x2 = self.S2(x)  # x1 x2 # 添加注意力来看解耦效果！！！！！！！！！！！！！！！！！！！！！！
        # print(x1.shape, x2.shape)
        # print("-------------x1,x2",x1,x2)
        x1 = self.cls_convs1(x1)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        #     # 回归=3x3conv（共享） + 3x3conv（共享） + 1x1pred
        x2 = self.reg_convs1(x2)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        #     # 置信度=3x3conv（共享）+ 3x3conv（共享） + 1x1pred
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):  # 检测头解耦和非解耦
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    # def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer 原来yolov5需要改动
    # def __init__(self, nc=80, anchors=(), ch=(), inplace=True, Decoupled=False):  #
    def __init__(self, nc=80, anchors=(), Decoupled=False, ch=(), inplace=True):  # 解耦
        super().__init__()
        self.decoupled = Decoupled  # 原来yolov5没有Decoupled=False！！！！ 需要改动！

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 需要改动！
        if self.decoupled:
            self.m = nn.ModuleList( DecoupledHead(ch[index], nc, 1, anchors, index) for index in range(len(ch)))  # yolox
            # ch[0]:ch[1]:ch[2]    ->  ch[0]:ch[1]:ch[2]虽然是不同的三个通道，但是最后对应的三个输入为32,16,8          通道：[128, 256, 512]
            # print("---m",ch)
        else:
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        # print( "detect:", ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # for i in x:
        #     print("x:",x, end=",")
        z = []  # inference output
        for i in range(self.nl):
            # print(len(x), len(self.m))
            x[i] = self.m[i](x[i])  # conv
            # print("----",x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


# 根据配置的.yaml文件搭建模型
class Model(nn.Module):
    # yolov5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name  # 获得xx.yaml文件名
            with open(cfg, encoding='ascii', errors='ignore') as f:  # 加载xx.yaml文件
                self.yaml = yaml.safe_load(f)  # model dict：存放刚才加载的xx.yaml文件

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels：给yaml文件添加一个关键字ch:3
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist:[4,6,10,14,17,20,23]
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # 取出模块的最后一次层：Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward：[8,16,32]
            m.anchors /= m.stride.view(-1, 1, 1)  # 获得的是特征层上anchor的大小
            check_anchor_order(m)  # 检测一下anchor的顺序是否正确
            self.stride = m.stride
            if not m.decoupled:
                self._initialize_biases()  # only run once  原来yolov5需要这里用的yolox的检测头因此不需要！！！ 需要改动

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs

        for m in self.model:
        # =============================
            # print("module: {}, {}".format(m, x.shape))  # 打印模块和每个模块的输入
        # ====================
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # if len(x) == 1:
            #     print("_forward_once的x", x.shape)
            # else:
            #     for x_temp in x:
            #         print("_forward_once的x", x_temp.shape)
            # if type(x) is list:
            #     for i in x:
            #         print(i.shape, end=",")
            #     print("\n")
            # else:
            #     print(x.shape)
            # print(m)
            # if len(x) == 1:
            #     print(x.shape)
            # else:
            #     for x_i in x:
            #         print(x_i.shape)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict:yolov5.yaml # , input_channels(3):[3]
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']  # gd:1,gw:1
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # 输出通道数：number of outputs = anchors * (classes + 5) = 255

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out：  ch=[3]  ch[-1] = 3
    # print("parse_model----",ch[-1])
    # layer用来存储下面创建的每一层，save是一个标签统计那些层的特征需要保存的，c2表示输出通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 举例第0层：from：-1, number：1, module：“Conv”, args：[64,6,2,2]
        # print("parse_model",f)
        m = eval(m) if isinstance(m, str) else m  # eval strings，m:<class'models.common.Conv'>
        for j, a in enumerate(args):  # 遍历args[64,6,2,2]参数
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings，[64,6,2,2]
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain：求n的实际值是多少

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 # BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, GML, GML_mask, Iden]:  # 判断这一层是什么结构        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, GML_mask, Iden]:  # 判断这一层是什么结构
            # print("ch[f]", len(ch), f)
            # print("m.name",m)
            # print("args[0]", args[0])
            # if isinstance(m, GML1):
            # print("args[0]----",args[0])
            # print(f)

            c1, c2 = ch[f], args[0]  # c1表示输入通道数，c2表示输出通道数

            # c1, c2 = ch[f], args[0]  # c1表示输入通道数，c2表示输出通道数
            # print("c1,c2",c1,c2)
            # if c2 != no and not (m in [GML, GML_mask, Iden, Hard, Easy]):  # if not output
            if c2 != no and not (m in [ GML_mask, Iden, Hard, Easy]):  # if not output
                c2 = make_divisible(c2 * gw, 8)  # 通道数一般设为8的倍数，这样GPU运算更加友好

            args = [c1, c2, *args[1:]]  # 重写Conv的输入参数  args[3,32,6,2,2]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [Hard, Easy]:
            args = [3] + args
            c2 = args[0]
        elif m in [Add, SKConv]:
            c2 = c1  # (ch[x] for x in f)[-1]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        # 添加小目标模块的有关代码
        elif m is space_to_depth:
            c2 = 4 * ch[f]

        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        if 515 in args:
            args[0] = 512
        # print(args)
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # print("m_",m_)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params：打印一些输出信息
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist：主要用来保存4层 6层的信息
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)  # [32],[32,64],[32,64,64]
    return nn.Sequential(*layers), sorted(save)  # [6,4,14,10,17,20,23] -> [4,6,10,14,17,20,23]


if __name__ == '__main__':
    # 定义一些参数信息
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # 以下所有内容,原作者已注释
    #     Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter('.')
    LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
