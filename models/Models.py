# Kaiyu Li
# https://github.com/likyoo
#

import torch.nn as nn
import torch

class conv_block_nested(nn.Module):    # 带残差的双卷积层
    '''改变通道，不改变尺寸'''
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):                # 2倍上采样
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)  # 2倍上采样，边界像素不变
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)  # 2 kernel_Size，2stride，width1 = 2*width


    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):  #CAM 通道注意力
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出大小为1的张量
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)  # 1×1卷积代替全连接
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)    # 输出与通道数量相同的几个加权概率值



class SNUNet_ECAM(nn.Module):                                             # 孪生Unet++  ECAM
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 32,64,128,256,512

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2倍下采样
        # 升维
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0]) #3-32
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])#32-64
        self.Up1_0 = up(filters[1]) # 上采样64通道
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])#64-128
        self.Up2_0 = up(filters[2]) # 上采样128通道
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])#128-256
        self.Up3_0 = up(filters[3]) # 上采样256通道
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])#256-512
        self.Up4_0 = up(filters[4]) # 上采样512通道
        # 降维
        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])#128-32
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])#256-64
        self.Up1_1 = up(filters[1])# 上采样64通道
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])#512-128
        self.Up2_1 = up(filters[2])# 上采样128通道
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])#1024-256
        self.Up3_1 = up(filters[3])# 上采样256通道

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)  # c=128
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4) # c=32

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1) # 128-2

        for m in self.modules():    # nn.Module类中的一个方法
            if isinstance(m, nn.Conv2d):  # 判断是否是nn.Conv2d类型
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # kaiming正态分布
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        # ---------------------------------------------        # xA、xB编码器
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))
        # ---------------------------------------------        # 各级解码器
        # L1解码器
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        # L2解码器
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))
        # L3解码器
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))
        # L4解码器
        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))
        # ---------------------------------------------       #  deep supervision
        # L1-L4 通道拼接
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        # ---------------------------------------------       #  ECAM 注意力
        # L1-L4 通道相加
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0) # stack在批量轴之前创建新维度并进行堆叠
        ca1 = self.ca1(intra)
        #
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))    # tensor.repeat(1，4，1，1)在通道维度重复
        out = self.conv_final(out)

        return (out, )


class Siam_NestedUNet_Conc(nn.Module):                       # 孪生Unet++
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))  # deep supervision
        return (output1, output2, output3, output4, output)

if __name__=='__main__':

    # import hiddenlayer as h
    # from graphviz import Source
    # model = Siam_NestedUNet_Conc(3, 2)
    # graph = h.build_graph(model, (torch.zeros([1, 3, 64, 64]), torch.zeros([1, 3, 64, 64])))  # 获取绘制图像的对象
    # graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
    # graph.layout_direction = 'TB'
    # graph.save("./hiddenlayer.png")  # 保存图像的路径


    # import netron
    # import onnx
    # model = Siam_NestedUNet_Conc(3, 1)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = x1
    # modelPath = "./demo.pth"
    # torch.onnx.export(model, (x1, x2), modelPath)
    # # 显示特征图维度
    # onnx_model = onnx.load(modelPath)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelPath)
    # # 启动netron
    # netron.start(modelPath)

    # from torchsummary import summary
    # model = Siam_NestedUNet_Conc(3, 2)
    # summary(model, input_size=[(1, 3, 64, 64), (1, 3, 64, 64)])

    # from torchviz import make_dot
    # # x = torch.randn(1, 1, 28, 28).requires_grad_(True)
    # y = model(x1,x2)
    # MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters())))
    # MyConvNetVis.format = "png"
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "./"
    # # # 生成文件
    # MyConvNetVis.view()

    model = Siam_NestedUNet_Conc(3, 2)
    cd_preds = model(x1, x2)  # 两通道预测结果(batch,2,H,W)
    cd_preds = cd_preds[-1]  # 提取一个通道作为预测结果(batch,1,H,W)
    print(cd_preds.shape)
    _, cd_preds = torch.max(cd_preds, 1)
    print(cd_preds.shape)