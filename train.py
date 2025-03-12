import torch
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.network import LVPNet
from torch import nn
import time
import os

import argparse
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder
import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.cuda.amp import GradScaler, autocast

import torch.nn.functional as F

import torchvision
print(torchvision.__version__)

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--pre_epochs', default=200, type=int, help='pre train epoch number')
parser.add_argument('--num_epochs', default=500000, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=1, type=int, help='train batch size')
parser.add_argument('--sub_rate', default=0.15, type=float, help='sampling sub rate')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
LOAD_EPOCH = 0
SECOND_STAGE = 800

# 创建训练数据集
train_set = TrainDatasetFromFolder(r'C:\Users\Jackel\Desktop\X\image_test', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)

# 初始化网络
net = LVPNet(BLOCK_SIZE, opt.sub_rate)

# 损失函数为多元交叉熵
mse_loss = nn.CrossEntropyLoss()

if opt.generatorWeights != '':
    net.load_state_dict(torch.load(opt.generatorWeights))
    LOAD_EPOCH = opt.loadEpoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    net = nn.DataParallel(net)
    net.to(device)
    mse_loss.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=1)

for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, 'loss1': 0, 'loss2': 0}

    net.train()
    scheduler.step()

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        target *= 255
        target = target
        target_one_hot = torch.zeros(batch_size, 1, 256, target.size(2), target.size(3), dtype=torch.uint8).cuda()

        index = target.long().unsqueeze(2).cuda()
        target_one_hot.scatter_(2, index, torch.ones_like(index, dtype=torch.uint8).cuda())
        target_one_hot = target_one_hot.float()  # 转换为浮点数

        real_img = Variable(target_one_hot)
        z = Variable(data)
        z.to(device)

        # 前向传播
        output, x_compensated, x_before = net(z, epoch)

        output = output.permute(0, 1, 3, 4, 2).contiguous()
        output = output.view(-1, 256)
        real_img = real_img.argmax(dim=2).contiguous()
        real_img = real_img.view(-1)

        # 计算损失
        loss1 = F.cross_entropy(output, real_img)
        loss2 = F.mse_loss(x_compensated, x_before)
        # if epoch < SECOND_STAGE:
        #     loss_total = loss1
        # else:
        #     loss_total = loss1 + 0.5 * loss2

        loss_total = loss1

        # 累加损失用于平均
        running_results['g_loss'] += loss_total.item() * batch_size
        running_results['loss1'] += loss1.item() * batch_size
        running_results['loss2'] += loss2.item() * batch_size

        # 打印当前 loss1 和 loss2
        train_bar.set_description(desc='[%d] Loss_Total: %.4f Loss1: %.4f Loss2: %.4f lr: %.7f' % (
            epoch, loss_total.item(), loss1.item(), loss2.item(), optimizer.param_groups[0]['lr']))

        # 反向传播
        optimizer.zero_grad()  # 清除旧的梯度
        loss_total.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 计算平均损失
    avg_loss_total = running_results['g_loss'] / running_results['batch_sizes']
    avg_loss1 = running_results['loss1'] / running_results['batch_sizes']
    avg_loss2 = running_results['loss2'] / running_results['batch_sizes']
    print(f"Epoch [{epoch}] Average Loss_Total: {avg_loss_total:.4f}, Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}")

    # 保存模型
    save_dir = 'epochs' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % 1 == 0:
        save_path = os.path.join(save_dir, f'net_epoch_{epoch}_{avg_loss_total:.6f}_{avg_loss1:.6f}_{avg_loss2:.6f}.pth')
        torch.save(net.state_dict(), save_path)