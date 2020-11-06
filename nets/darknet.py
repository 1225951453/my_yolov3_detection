import torch
import torch.nn as nn
import math
from collections import OrderedDict
import os

# def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
#     for i in range(len(out_pool_size)):
#         # print(previous_conv_size)
#         h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
#         w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
#         h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
#         w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
#         maxpool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
#         x = maxpool(previous_conv)
#         if (i == 0):
#             spp = x.view(num_sample, -1)
#             # print("spp size:",spp.size())
#         else:
#             # print("size:",spp.size())
#             spp = torch.cat((spp, x.view(num_sample, -1)), 1)
#     return spp



class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResidualBlock, self).__init__()
        self.output_num = [16,4,1]
        self.output_channels = planes[1]
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes[0])
        self.bn1 = nn.GroupNorm(16, planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes[1])
        self.bn2 = nn.GroupNorm(16, planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

#         spp1 = spatial_pyramid_pool(out,1, [int(out.size(2)), int(out.size(3))],self.output_num)
#         spp2 = spatial_pyramid_pool(x,1, [int(x.size(2)), int(x.size(3))],self.output_num)
#         out = torch.cat((spp1, spp2), 1)
#         out = out.view(8,-1,16,16)
#         out = out.reshape(out.size(1),-1,0,0)
#         t_conv = nn.ConvTranspose2d(in_channels = out.shape[1],out_channels = (128,128), kernel_size = (1,1), stride=1)
#         conv = nn.Conv2d(out.size(1), self.output_channels, kernel_size=(3, 3),padding = 1)
#         out = t_conv(out)
        out += residual
        return out


###搭网络，初始化权重
class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.bn1 = nn.GroupNorm(16, self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 为啥要这么初始化卷积核参数呢？？？？
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #                 torch.nn.init.xavier_normal_(m.weight, gain=1.0)
#                 torch.nn.init.kaiming_normal_(m.state_dict(), a=1, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m,nn.GroupNorm ): #nn.BatchNorm2d
                # 为啥BN的权重都为1？？？？？？？
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #                 torch.nn.init.xavier_normal_(m.bias, gain=1.0)
                #                 torch.nn.init.xavier_normal_(m.weight, gain=1.0)
#                 torch.nn.init.kaiming_normal_(m.state_dict(), a=1, mode='fan_in', nonlinearity='leaky_relu')

    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
#         layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_bn", nn.GroupNorm(16, planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入darknet模块
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), ResidualBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model

