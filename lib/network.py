from torch import nn

import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import heapq
from collections import defaultdict
import torch
from torch.autograd import Function


# Reshape + Concat layer

class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


# # 量化函数：根据 Step 量化输入
# def quantize(x, step_size):
#     # 使用 floor 函数进行量化
#     return torch.floor(x / step_size)
#
#
# # 反量化函数：根据 Step 将量化值转换为原始范围
# def dequantize(x, step_size):
#     return x * step_size

class STEQuantizer(Function):
    @staticmethod
    def forward(ctx, input_, step):
        """
        前向传播：量化操作
        """
        ctx.save_for_backward(input_, step)
        return torch.floor(input_ / step)  # 向下取整实现量化

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：STE策略
        """
        input_, step = ctx.saved_tensors
        grad_input = grad_output.clone()  # 使用STE直接传递梯度
        return grad_input, None  # 对step的梯度可以忽略


def ste_quantize(input_, step):
    return STEQuantizer.apply(input_, step)


def dequantize(input_, step):
    """
    反量化操作：将量化后的整数还原为浮点数
    """
    return input_ * step


# 熵编码函数：简单的频率统计和概率模型
def entropy_encode(quantized_values):
    unique_vals, counts = torch.unique(quantized_values, return_counts=True)
    probs = counts.float() / quantized_values.numel()

    encoding = {}
    cumulative_prob = torch.cumsum(probs, dim=0)
    for i, val in enumerate(unique_vals):
        encoding[val.item()] = cumulative_prob[i].item()  # 使用累积概率表示编码

    return encoding


# 熵解码函数：使用累积概率恢复量化值
def entropy_decode(encoding, quantized_values):
    decoded_values = []
    for val in quantized_values:
        for key, prob in encoding.items():
            if prob >= val:
                decoded_values.append(key)
                break
    return torch.tensor(decoded_values)


class EntropyCoder:
    def __init__(self):
        self.prob_model = None  # 概率模型，用于编码/解码

    def encode(self, data):
        """
        熵编码过程：假设使用简单的频率模型。
        :param data: 已经量化后的测量值（整数值）。
        :return: 编码后的比特流
        """
        # 使用直方图模拟概率模型
        unique, counts = torch.unique(data, return_counts=True)
        self.prob_model = dict(zip(unique.tolist(), counts.tolist()))

        # 简单编码（模拟比特流）
        bitstream = []
        for value in data.flatten():
            temp = bin(int(value.item()))
            index = temp.find('b')
            bitstream.append(temp[index + 1:].zfill(8))
        return ''.join(bitstream)

    def decode(self, bitstream, shape):
        """
        熵解码过程：还原数据。
        :param bitstream: 编码后的比特流。
        :param shape: 数据的原始形状。
        :return: 解码后的量化值。
        """
        # 将比特流还原为整数
        values = [int(bitstream[i:i + 8], 2) for i in range(0, len(bitstream), 8)]
        return torch.tensor(values).reshape(shape)


class EntropyCoder_Huffman:
    def __init__(self):
        self.huffman_tree = None  # 哈夫曼树，用于编码/解码
        self.encoding_table = None  # 符号到编码的映射表
        self.decoding_table = None  # 编码到符号的映射表

    def _build_huffman_tree(self, data):
        """
        构建哈夫曼树。
        :param data: 输入数据，用于统计频率。
        """
        freq = defaultdict(int)
        for value in data.flatten():
            freq[int(value.item())] += 1

        # 使用优先队列（最小堆）构建哈夫曼树
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            low = heapq.heappop(heap)
            high = heapq.heappop(heap)
            for pair in low[1:]:
                pair[1] = '0' + pair[1]
            for pair in high[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])

        # 构建映射表
        huffman_tree = heapq.heappop(heap)[1:]
        self.encoding_table = {symbol: code for symbol, code in huffman_tree}
        self.decoding_table = {code: symbol for symbol, code in self.encoding_table.items()}

    def encode(self, data):
        """
        哈夫曼编码过程。
        :param data: 已经量化后的测量值（整数值）。
        :return: 编码后的比特流
        """
        # 构建哈夫曼树并生成编码表
        self._build_huffman_tree(data)

        # 使用编码表将数据编码为比特流
        bitstream = ''.join(self.encoding_table[int(value.item())] for value in data.flatten())
        return bitstream

    def decode(self, bitstream, shape):
        """
        哈夫曼解码过程。
        :param bitstream: 编码后的比特流。
        :param shape: 数据的原始形状。
        :return: 解码后的量化值。
        """
        # 使用解码表还原数据
        decoded_values = []
        current_code = ""
        for bit in bitstream:
            current_code += bit
            if current_code in self.decoding_table:
                decoded_values.append(self.decoding_table[current_code])
                current_code = ""

        return torch.tensor(decoded_values).reshape(shape)


class QuantizationCompensationModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(QuantizationCompensationModule, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(154, 154, kernel_size=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(173)

        # 第二层卷积
        self.conv2 = nn.Conv2d(154, 154, kernel_size=1, padding=0)
        # self.bn2 = nn.BatchNorm2d(173)

        # 第三层卷积
        self.conv3 = nn.Conv2d(154, 154, kernel_size=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 第一层
        residual = x  # 残差连接
        x = F.relu(self.conv1(x))
        x = x + residual  # 添加残差连接

        # 第二层
        residual = x  # 残差连接
        x = F.relu(self.conv2(x))
        x = x + residual  # 添加残差连接

        # 第三层
        residual = x  # 残差连接
        x = F.relu(self.conv3(x))
        x = x + residual  # 添加残差连接

        return x


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.ReLU()
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.ReLU()
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1_1 = self.conv1_1(x)
        f3_1 = self.conv3_1(x)
        f5_1 = self.conv5_1(x)
        f7_1 = self.conv7_1(x)
        f2 = f1_1 + f3_1 + f5_1 + f7_1
        f1_2 = self.conv1_2(f2)
        f3_2 = self.conv3_2(f2)
        f5_2 = self.conv5_2(f2)
        f7_2 = self.conv7_2(f2)
        f3 = f1_2 + f3_2 + f5_2 + f7_2 + x
        f4 = self.conv_fusion(f3) * f3 + f3
        return f4


class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureEnhancement, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv1_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.fusion = nn.ReLU()

    def forward(self, x):
        f3 = self.conv3(x)
        f3_1 = self.conv3_1(f3)
        f1 = self.conv1(x)
        f_fusion = self.fusion(f3_1 + f1)
        return self.conv1_fusion(f_fusion)

def New_Reshape(input_tensor):
    # 将64个通道展开为8x8的块，每个块对应一个16x16的区域
    # 这一步将64拆成8行8列的块
    output_tensor = input_tensor.view(-1, 8, 8, 128, 128)

    # 接下来，我们需要将这些块填入对应的区域
    # 我们通过拼接将它们放到16x16的网格中

    # 第一步：将 (batch_size, 8, 8, 16, 16) 展开为 (batch_size, 8, 8, 16, 16)
    output_tensor = output_tensor.permute(0, 1, 3, 2, 4)  # 交换维度，使其变为 (batch_size, 8, 16, 8, 16)

    # 第二步：将每个块放置到 128x128 的空间中
    # 需要将这些块按照正确的顺序拼接起来
    output_tensor = output_tensor.contiguous().view(-1, 1024, 1024)

    return output_tensor.unsqueeze(1)

# LVPNet 网络的更新版本
class LVPNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.15, step_size=0.01):
        super(LVPNet, self).__init__()
        self.blocksize = blocksize
        self.step_size = step_size

        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize * blocksize * subrate)), blocksize, stride=blocksize,
                                  padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, 1, stride=1,
                                    padding=0)

        # reconstruction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 256, kernel_size=3, padding=1)

        self.qcm_module = QuantizationCompensationModule()

        self.MSFF = nn.ModuleList(
            [MultiScaleFeatureFusion(1, 1) for _ in range(3)]
        )

        # self.FE = nn.ModuleList(
        #     [FeatureEnhancement(1, 1), FeatureEnhancement(1, 1), FeatureEnhancement(1, 1),
        #      FeatureEnhancement(36, 36), FeatureEnhancement(36, 36),
        #      FeatureEnhancement(72, 72)]
        # )
        #
        # self.PD1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 36, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU()
        # )
        #
        # self.PD1_1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 36, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU()
        # )
        #
        # self.PD2 = nn.Sequential(
        #     nn.Conv2d(36, 72, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU(),
        #     # nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=0, bias=False),
        #     # nn.ReLU()
        # )
        #
        # self.PD2_2 = nn.Sequential(
        #     nn.Conv2d(36, 72, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU(),
        #     # nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=0, bias=False),
        #     # nn.ReLU()
        # )
        #
        # self.PD3 = nn.Sequential(
        #     nn.Conv2d(72, 173, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU()
        # )
        #
        # self.PD3_3 = nn.Sequential(
        #     nn.Conv2d(72, 173, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.ReLU()
        # )

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, padding=0, bias=False),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 256, kernel_size=4, stride=4, padding=0, bias=False),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(1024, 154, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.upsampling2 = nn.Conv2d(16, 64, 1, stride=1,
                                    padding=0)

    def forward(self, x, epoch):
        x = x.to('cuda')
        # 采样
        # x_before = self.sampling(x)

        # x4 = self.sampling(x)

        # x1 = self.MSFF[0](x)
        # x1_1 = self.FE[0](x1)
        # # x1_2 = self.FE[1](x1_1)
        # # x1_3 = self.FE[2](x1_2)
        # x1d1 = self.PD1(x1 + x)
        # x1_3d = self.PD1_1(x1_1)
        # x2 = self.MSFF[1](x1d1)
        # x2_1 = self.FE[3](x2)
        # # x2_2 = self.FE[4](x2_1)
        # x2d1 = self.PD2(x2 + x1d1)
        # x2_2d = self.PD2_2(x1_3d + x2_1)
        # x3 = self.MSFF[2](x2d1)
        # x3_1 = self.FE[5](x3)
        # x3d1 = self.PD3(x3 + x2d1)
        # x3_1d = self.PD3_3(x2_2d + x3_1)
        # x4 = self.MSFF[3](x3d1 + x3_1d)

        # x1 = self.MSFF[0](x)
        # x1_1 = self.FE[0](x1)
        # x1_2 = self.FE[1](x1_1)
        # x1_3 = self.FE[2](x1_2)
        # x1d1 = self.PD1(x1)
        # x1_3d = self.PD1_1(x1_3)
        # x2 = self.MSFF[1](x1d1)
        # x2_1 = self.FE[3](x2)
        # x2_2 = self.FE[4](x2_1)
        # x2d1 = self.PD2(x2)
        # x2_2d = self.PD2_2(x1_3d + x2_2)
        # x3 = self.MSFF[2](x2d1)
        # x3_1 = self.FE[5](x3)
        # x3d1 = self.PD3(x3)
        # x3_1d = self.PD3_3(x2_2d + x3_1)
        # x4 = self.MSFF[3](x3d1 + x3_1d)

        # for i in range(3):
        #     x = self.MSFF[i](x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x4 = self.down4(x)

        # x2 = self.PD1(x)
        # x3 = self.PD2(x2)
        # x4 = self.PD3(x3)

        # x_before = torch.sigmoid(x)

        # 量化
        # x_quantized = torch.round(x / self.step_size)

        # # 加入量化操作
        quantized_x = ste_quantize(x4, torch.tensor(self.step_size, device=x4.device))

        # 加入反量化操作
        x_reconstructed = dequantize(quantized_x, torch.tensor(self.step_size, device=x.device))

        # 熵编码
        # coder = EntropyCoder_Huffman()
        # bitstream = coder.encode(x_quantized)
        # print(len(bitstream))
        #
        # 熵解码
        # x_decoded = coder.decode(bitstream, x_quantized.shape)

        # 反量化
        # x_reconstructed = x_decoded * self.step_size

        # x_reconstructed = x_quantized * self.step_size

        x_reconstructed = x_reconstructed

        x_res = self.qcm_module(x_reconstructed)

        # if epoch < 800:
        #     x_compensated = x_reconstructed
        # else:
        #     x_compensated = x_res + x_reconstructed

        x_compensated = x_res + x_reconstructed

        # 上采样
        # x = self.upsampling2(x_compensated)

        x = self.upsampling(x_compensated)
        x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat
        # x = My_Reshape_Adap(x, 8)  # Reshape + Concat
        # x = New_Reshape(x)

        # 重建网络
        block1 = self.conv1(x)
        block2 = self.conv2(block1 + x)
        block3 = self.conv3(block2 + block1)
        block4 = self.conv4(block3 + block2)
        block5 = self.conv5(block4 + block3)

        return block5.view(block5.size(0), 1, 256, block5.size(2), block5.size(3)), x_compensated, x4
