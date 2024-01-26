import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn

class HSVHotMap(nn.Module):
    def __init__(self, lower_red=[0, 110, 110], upper_red=[14, 255, 255], eps=1e-8, method_name = "hot"):
        super(HSVHotMap, self).__init__()
        self.eps = eps
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.kernel = 10
        self.dilate = nn.MaxPool2d(self.kernel, stride=1)
        self.kernel -= 1
        self.guass_filiter = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        )
        self.name = method_name # "mask", "hot"
        # 高斯滤波
        w1 = torch.Tensor(
            np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]).reshape(1, 1, 3, 3))
        self.guass_filiter[0].weight = nn.Parameter(w1)

    def rgb_to_hsv(self, img):

        hue = torch.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype = img.dtype).to(img.device)
        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6
        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0
        value = img.max(1)[0]
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def filter_out_red_torch(self, torch_image):
        hsv_back = torch_image.permute(0, 2, 3, 1) * 255
        mask = (hsv_back[:, :, :, 0] >= self.lower_red[0]) * (hsv_back[:, :, :, 1] >= self.lower_red[1]) * (
                    hsv_back[:, :, :, 2] >= self.lower_red[2])
        mask2 = (hsv_back[:, :, :, 0] < self.upper_red[0]) * (hsv_back[:, :, :, 1] <= self.upper_red[1]) * (
                    hsv_back[:, :, :, 2] <= self.upper_red[2])

        #         mask = (hsv_back[:,:,:,0] >= 0) * (hsv_back[:,:,:,1] >= 110) * (hsv_back[:,:,:,2] >= 110)
        #         mask2 = (hsv_back[:,:,:,0] < 14) * (hsv_back[:, :,:,1] <= 255) * (hsv_back[:,:,:,2] <= 255)

        mask *= mask2
        # print(hsv_back.shape, mask.shape)
        return mask.to(torch_image.dtype)

    def build_mask(self, x):
        # print("in:", x.shape)
        x = self.rgb_to_hsv(x)
        # print("filter_out_red_torch:", x.shape)
        mask = self.filter_out_red_torch(x)
        return mask.unsqueeze(1)

    @torch.no_grad()
    def guass_process(self, x):
        return self.guass_filiter(x)

    def build_hotmap(self, x):
        x = self.rgb_to_hsv(x)
        mask = self.filter_out_red_torch(x)
        # mask = mask * 0.5 + torch.ones_like(mask) * 0.5
        hotmap = self.guass_process(mask.unsqueeze(1))
        return hotmap

    @torch.no_grad()
    def show(self, origin, mask):
        origin = origin.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        plt.figure(figsize=(12, 24))
        plt.subplot(311)
        plt.imshow(mask.squeeze().numpy())
        plt.subplot(312)
        a = (origin * mask.repeat(1, 1, 3))
        plt.imshow(np.array(255 * a.squeeze().numpy(), dtype=np.uint8))
        plt.subplot(313)
        plt.imshow(origin.squeeze().numpy())
        plt.show()

    def forward(self, x):
        if self.name == "hot":
            return self.build_hotmap(x)
        elif self.name == "mask":
            return self.build_mask(x)
        else:
            raise NotImplementedError




from .former import MHSA


class GML_mask(nn.Module): # 以这个GML为基准 通过控制method_name来选择hot和mask
    def __init__(self, in_dim, out_dim = 3):
        super().__init__()
        self.gethotmap = HSVHotMap( method_name = "hot")
        self.SHOW = 3
        # self.USE = False  # 用来控制是否使用mask，不管mask还是hot！
        self.USE = True  # 用来控制是否使用mask，不管mask还是hot！

    def get_guide(self, x):
        # print("----GML in ", x.shape,x) # [16, 3, 640, 640] x为0-1的值
        x = x.clone().detach()
        attn_hotmap = self.gethotmap(x)
        # print("x 经过hsv转换",attn_hotmap.shape,attn_hotmap)  # [16, 1, 640, 640] attn_hotmap为0和[0-1]的有效值
        # print("---",torch.max(attn_hotmap,dim =-1))
        x_mask = attn_hotmap # * x

        # print("--------------", x_mask.shape)
        # if (self.SHOW):
        #     # print(x.shape)
        #     if len(x) != 1:
        #         for i in range(5):
        #             test_img = np.array(255 * x_mask[i].squeeze().permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
        #             test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        #             cv2.imwrite("./MASK_{}.png".format(i), test_img)
        #     self.SHOW -= 1
        # if (self.SHOW):
        #     print("----GML out", x_mask.shape)
        #     self.SHOW -= 1
        return x_mask

    @torch.no_grad()
    def forward(self, x):
        # print("-----------",x.shape)

        # print("GML_mask的 输出结果",x_mask.shape,x_mask) # [1, 1, 640, 640] x_mask为0和[0-1]的有效值
        if self.USE:
            x_mask = self.get_guide(x)
            return x_mask
        else:
            return torch.zeros_like(x)


class Hard(nn.Module):
    def __init__(self, in_dim, out_dim = 3, k=1, s=1, p=0):
        super().__init__()

    def checkmask(self, x): # 这里的check好像没用！！！
        a = torch.sum(x, [2, 3])
        a[a > 0] = 1
        d = torch.ones_like(x)
        return torch.ones_like(x) - (a.unsqueeze(-1).unsqueeze(-1) * d) + x

    @torch.no_grad()
    def forward(self, y):
        mask, x = y
        hard = x # - x * mask
        return hard

class resto(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.c0 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 2,1,1, dilation=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, 2,1,1, dilation=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, 2,1,1, dilation=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.c3 = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),

        )
        self.c4 = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),
        )
        self.sigmod = nn.ReLU()
    def forward(self, x):
        x = self.c0(x)
        ans1= self.c1(x)
        ans2 = self.c2(x)
        x_temp = self.sigmod(ans1 * ans2)
        shot1 = self.c3( x_temp * x )
        return shot1

class formerblock(nn.Module):
        def __init__(self, in_dim):
            s_out_dim = in_dim
            super().__init__()
            self.encode = MHSA(
                in_channels=s_out_dim,
                heads=4, curr_h=4, curr_w=4,
                pos_enc_type="relative" #相对位置编码的模式
            )
        def forward(self, easy_feature):
            # print("++++easy_feature",easy_feature.shape)
            Q_h = Q_w = 4 # 每个区域的边长
            # print("====easy_feature",easy_feature.shape)
            N, C, H, W = easy_feature.shape
            # print("easy in:", easy_feature.shape)
            P_h, P_w = H // Q_h, W // Q_w # 个数
            # print(P_h, P_w)
            # print("---easy_feature",easy_feature.shape)
            easy_feature = easy_feature.reshape(N * P_h * P_w, C, Q_h, Q_w) #
            # print("++++easy_feature",easy_feature.shape)
            # print("easy:", easy_feature.shape)
            easy_feature = self.encode(easy_feature)
            easy_feature = easy_feature.permute(0, 3, 1, 2)  # back to pytorch dim order ，标准的dim序列是（N, C, H, W）
            N1, C1, H1, W1 = easy_feature.shape
            easy_feature = easy_feature.reshape(N, C1, int(H), int(W))
            return easy_feature

class double(nn.Module): # 普通卷积和空洞卷积双支路block
    def __init__(self, in_dim, out_dim, s = [2,2], k = [3, 2], p = [2, 0]):
        super(double, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, k[0], s[0], p[0], dilation = 2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, s[1], k[1], p[1]),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.act = nn.Sigmoid() # 为什么要进行sigmoid函数？？
        self.c3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        # self.down = nn.Upsample(scale_factor=0.5)
    def forward(self, x):
        # x = self.down(x)
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3( self.act(x1+x2))
        return x3

class Easy(nn.Module):  # 64
    def __init__(self, in_dim, out_dim = 128, k=1, s=1, iter=0, p=0): # 默认sppf前放置，则out_dim = 128
        super().__init__()
        s_out_dim = out_dim // 8

        d = [
            double(3, s_out_dim)
            # nn.Conv2d(3, s_out_dim, k, s, p),
            # nn.BatchNorm2d(s_out_dim),
            # nn.ReLU(),
        ]
        if iter != 0 or iter != 1:
            for i in range(iter - 1):
                d.append(double( s_out_dim, s_out_dim )) # image encoder

        self.encode = formerblock(s_out_dim) # resto( s_out_dim, s_out_dim) #
        self.encode_conv = resto( s_out_dim, s_out_dim)

        self.select = nn.Sequential(*d)

        self.last_act = nn.Sequential(
            nn.Conv2d(2 * s_out_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.kernel = 10
        self.dilate = nn.MaxPool2d(self.kernel, stride = 1)
        self.kernel = 9

    def checkmask(self, x):
        # print("-----------",x.shape) # [16,1,640,640]
        a = torch.sum(x, [2, 3]) # [16, 1]
        a[a > 0] = 1
        d = torch.ones_like(x)
        return torch.ones_like(x) - (a.unsqueeze(-1).unsqueeze(-1) * d) + x
        # print("-----------",a.shape)
        # print(a)
    # print(torch.max(a.unsqueeze(-1).unsqueeze(-1)))
    # print("-----------")
    # print(a.unsqueeze(-1).unsqueeze(-1)* d)
    # print(torch.ones_like(x) - (a.unsqueeze(-1).unsqueeze(-1) * d) + x)

    def forward(self, y):
        mask, x = y  # mask是GML所得到的掩码
        # print("x",x.shape,x) # x全是[0,1]的值 [16, 3, 640, 640]
        # print("mask",mask.shape,torch.max(mask,dim=-1)) #  [16, 1, 640, 640] mask为0和非0[其中非0值范围在0到1]
        # easy_x = x
        # easy_x = x * mask  # 全黑，即：全是无效的mask self.USE = False
        easy_x = x * self.checkmask(mask) # 有效的mask
        # print('easy_x',easy_x.shape,easy_x) # [16, 3, 640, 640] 有0有非0[其中非0值范围在0到1]

        easy_feature_origin = self.select(easy_x) # 通过【空洞卷积核正常卷积并联的block】*N个
        easy_feature = self.encode( easy_feature_origin )  # formerblock
        easy_feature_ = self.encode_conv(easy_feature_origin)  # resto
        # print("out:", easy_feature.shape)
        easy_feature = self.last_act( torch.cat([easy_feature, easy_feature_ ], 1)) # se attention * x # 进行self.last_act的1x1卷积块

        return easy_feature