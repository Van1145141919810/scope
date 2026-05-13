#!/usr/bin/env python
"""
SCOPE 项目 — 训练脚本
======================
实现 β-VAE 训练循环。

损失函数: L = BCE(prediction, ground_truth) + β × KL_divergence
          β = 0.01（KL 权重）

β-VAE 原理:
  - BCE: 重建损失 — 预测的占用图与真实值有多大差异
  - KL:  正则化损失 — latent 分布与标准正态分布有多大差异
  - β(0.01): 小的 KL 权重 → 优先重建精度，允许 latent 空间有更大灵活性

训练策略:
  - 监督信号只用 mask[:, 0]（未来第1帧）
  - 使用 Adam 优化器
  - 每 10 epoch 保存一次 checkpoint
  - 支持断点续训（加载已有模型继续训练）

用法:
  python train.py <model_path> <train_dir> <val_dir>
  例: python train.py ./model/model.pth ~/OGM-datasets/OGM-Turtlebot2/train ~/OGM-datasets/OGM-Turtlebot2/val
"""

# ---- 导入 ----
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm                               # 进度条

from tensorboardX import SummaryWriter              # TensorBoard 日志（注意：tensorboardX，非标准 tensorboard）
import numpy as np

# 从同目录导入模型和数据加载
from model import *                                  # scope模型 + VaeTestDataset + 所有常量/工具函数
from local_occ_grid_map import LocalMap              # GPU 并行建图

import sys
import os


# =============================================================================
# 全局配置
# =============================================================================

# 默认模型存储路径
model_dir = './model/model.pth'
NUM_ARGS = 3                  # 命令行参数数量: [script, model_path, train_dir, val_dir]

# 训练超参数
NUM_EPOCHS = 50              # 训练总轮数（原论文 50 轮）
BATCH_SIZE = 128             # 批大小（原论文注释中提到可调至 512）
LEARNING_RATE = "lr"         # optimizer param dict 的 key
BETAS = "betas"              # Adam β 参数
EPS = "eps"                  # Adam ε 参数
WEIGHT_DECAY = "weight_decay"  # L2 正则化系数

# 模型常量
NUM_INPUT_CHANNELS = 1       # 输入通道数（单通道二值栅格）
NUM_LATENT_DIM = 512         # VAE latent 维度 (2×16×16)
NUM_OUTPUT_CHANNELS = 1      # 输出通道数（单通道占用概率）
BETA = 0.01                  # ★ β-VAE 的 β：KL 损失权重（越小→越看重重建精度）

# 建图参数
P_prior = 0.5                # 先验占用概率 50%（完全未知）
P_occ = 0.7                  # 占用置信概率（击中时更新到 0.7）
P_free = 0.3                 # 空闲置信概率（穿过时更新到 0.3）
MAP_X_LIMIT = [0, 6.4]       # x 轴范围（0 ~ 6.4 米）
MAP_Y_LIMIT = [-3.2, 3.2]    # y 轴范围（-3.2 ~ 3.2 米）
RESOLUTION = 0.1             # 栅格分辨率（0.1 米/格 = 64×64）
TRESHOLD_P_OCC = 0.8          # 占用判定阈值（p ≥ 0.8 → 占用）

# 设置随机种子（保证可复现）
set_seed(SEED1)


# =============================================================================
# 学习率调度器
# =============================================================================

def adjust_learning_rate(optimizer, epoch):
    """
    阶梯式学习率调整。

    策略（原论文设定）:
        epoch ≤ 30000:   lr = 1e-4
        30000 < epoch ≤ 50000: lr = 3e-4   ← 提高！让模型跳出局部最优
        48000 < epoch ≤ 8300: lr = 2e-5    ← 注意：48000有bug（应该是50000之后），实际可能不触发
        epoch > 48000:  lr = lr * 0.1^(epoch//110000)

    注意: 本项目 NUM_EPOCHS=50，实际始终使用 lr=1e-4。
          这些阶梯是原论文作者为长时间训练准备的（可达数万轮）。
    """
    lr = 1e-4
    if epoch > 30000:
        lr = 3e-4
    if epoch > 50000:
        lr = 2e-5
    if epoch > 48000:
        lr = lr * (0.1 ** (epoch // 110000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# =============================================================================
# 训练函数
# =============================================================================

def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    """
    单轮训练循环。

    步骤:
        1. 遍历所有 batch
        2. 对每个样本: LiDAR → 构建输入栅格 + 目标栅格
        3. 前向传播 → 计算 β-VAE 损失
        4. 反向传播 → 更新权重

    返回:
        train_loss:    总损失平均值
        train_kl_loss: KL 散度平均值
        train_ce_loss: 交叉熵平均值
    """
    model.train()  # 切换到训练模式（启用 BN、Dropout 等）

    running_loss = 0.0    # 累计总损失
    kl_avg_loss = 0.0     # 累计 KL 损失
    ce_avg_loss = 0.0     # 累计 BCE 损失
    counter = 0            # batch 计数器

    # 计算总 batch 数（向上取整）
    num_batches = int(len(dataset) / dataloader.batch_size)

    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1

        # ---- 提取数据 ----
        # scans:     (128, 20, 1080) — 前10帧=历史，后10帧=未来
        # positions: (128, 20, 3)     — [x, y, θ]
        # velocities:(128, 20, 2)     — [v_linear, v_angular]
        scans = batch['scan'].to(device)
        positions = batch['position'].to(device)
        velocities = batch['velocity'].to(device)

        batch_size = scans.size(0)  # 128

        # =====================================================================
        # Step A: 构建目标栅格（mask）—— 未来 10 帧的真值
        # =====================================================================
        # LiDAR 角度: -135° ~ 135°，共 1080 个角度（均匀分布）
        angles = torch.linspace(
            -(135 * np.pi / 180), 135 * np.pi / 180,
            scans[:, SEQ_LEN:].shape[-1]  # 1080
        ).to(device)

        # 机器人历史位姿置零（用于 gt mask 计算）
        x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)

        # 取后 10 帧 LiDAR（未来帧）
        distances = scans[:, SEQ_LEN:]          # (128, 10, 1080)

        # 建图：极坐标→笛卡尔
        distances_x, distances_y = mask_gridMap.lidar_scan_xy(
            distances, angles, x_odom, y_odom, theta_odom
        )
        # 离散化：→ 二值栅格 (128, 10, 64, 64)
        mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)

        # =====================================================================
        # Step B: 构建输入栅格 —— 过去 10 帧
        # =====================================================================
        input_gridMap = LocalMap(
            X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT,
            resolution=RESOLUTION, p=P_prior,
            size=[batch_size, SEQ_LEN], device=device
        )

        # B1. 预测未来参考系原点（1 步后）
        obs_pos_N = positions[:, SEQ_LEN - 1]  # 当前位姿（第10帧）
        vel_N = velocities[:, SEQ_LEN - 1]     # 当前速度
        T = 1                                    # 预测 1 步
        noise_std = [0, 0, 0]                   # 训练时不加噪声
        pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)

        # B2. 坐标变换：过去位姿 → 未来参考系
        pos = positions[:, :SEQ_LEN]            # (128, 10, 3) 前10帧位姿
        x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)

        # B3. 取前 10 帧 LiDAR
        distances = scans[:, :SEQ_LEN]          # (128, 10, 1080)

        # B4. 建图
        distances_x, distances_y = input_gridMap.lidar_scan_xy(
            distances, angles, x_odom, y_odom, theta_odom
        )
        input_binary_maps = input_gridMap.discretize(distances_x, distances_y)

        # =====================================================================
        # Step C: 添加 channel 维度 + 前向传播
        # =====================================================================
        # unsqueeze(2): (128, 10, 64, 64) → (128, 10, 1, 64, 64)
        input_binary_maps = input_binary_maps.unsqueeze(2)
        mask_binary_maps = mask_binary_maps.unsqueeze(2)

        # 梯度清零（每 batch 必须重置）
        optimizer.zero_grad()

        # ★ 前向传播
        prediction, kl_loss = model(input_binary_maps)

        # =====================================================================
        # Step D: 损失计算
        # =====================================================================
        # BCE 损失: 预测 vs 真实未来第1帧
        # mask_binary_maps[:, 0]: 只取未来第 1 帧做监督！
        # reduction='sum' → 总和（除以 batch_size 做平均）
        ce_loss = criterion(prediction, mask_binary_maps[:, 0]).div(batch_size)

        # β-VAE 损失 = 重建误差 + β × KL 正则化
        loss = ce_loss + BETA * kl_loss

        # =====================================================================
        # Step E: 反向传播
        # =====================================================================
        # torch.ones_like(loss): 损失标量的梯度权重=1
        # 多 GPU 时需要此参数来正确处理 DataParallel
        loss.backward(torch.ones_like(loss))
        optimizer.step()

        # =====================================================================
        # Step F: 累积统计
        # =====================================================================
        # 多 GPU 时取均值（DataParallel 返回的 loss 是各 GPU 结果的列表）
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            ce_loss = ce_loss.mean()
            kl_loss = kl_loss.mean()

        running_loss += loss.item()
        kl_avg_loss += kl_loss.item()
        ce_avg_loss += ce_loss.item()

        # 每 128 batch 打印一次
        if i % 128 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
                  .format(epoch, epochs, i + 1, num_batches,
                          loss.item(), ce_loss.item(), kl_loss.item()))

    # 本轮平均损失
    train_loss = running_loss / counter
    train_kl_loss = kl_avg_loss / counter
    train_ce_loss = ce_avg_loss / counter

    return train_loss, train_kl_loss, train_ce_loss


# =============================================================================
# 验证函数
# =============================================================================

def validate(model, dataloader, dataset, device, criterion):
    """
    单轮验证循环 —— 与 train() 逻辑几乎相同，但：
      - model.eval() + torch.no_grad(): 禁用梯度计算 → 快且省显存
      - 不做反向传播和参数更新

    返回:
        val_loss, val_kl_loss, val_ce_loss
    """
    model.eval()  # 切换到评估模式

    running_loss = 0.0
    kl_avg_loss = 0.0
    ce_avg_loss = 0.0
    counter = 0

    num_batches = int(len(dataset) / dataloader.batch_size)

    with torch.no_grad():  # ★ 关键: 不计算梯度（省显存，加速推理）
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1

            # ---- 提取数据 ----
            scans = batch['scan'].to(device)
            positions = batch['position'].to(device)
            velocities = batch['velocity'].to(device)
            batch_size = scans.size(0)

            # ---- 构建目标栅格 (mask) ----
            mask_gridMap = LocalMap(
                X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT,
                resolution=RESOLUTION, p=P_prior,
                size=[batch_size, SEQ_LEN], device=device
            )
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            distances = scans[:, SEQ_LEN:]
            angles = torch.linspace(-(135 * np.pi / 180), 135 * np.pi / 180,
                                     distances.shape[-1]).to(device)
            distances_x, distances_y = mask_gridMap.lidar_scan_xy(
                distances, angles, x_odom, y_odom, theta_odom
            )
            mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)

            # ---- 构建输入栅格 ----
            input_gridMap = LocalMap(
                X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT,
                resolution=RESOLUTION, p=P_prior,
                size=[batch_size, SEQ_LEN], device=device
            )
            obs_pos_N = positions[:, SEQ_LEN - 1]
            vel_N = velocities[:, SEQ_LEN - 1]
            T = 1
            noise_std = [0, 0, 0]
            pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
            pos = positions[:, :SEQ_LEN]
            x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)
            distances = scans[:, :SEQ_LEN]
            distances_x, distances_y = input_gridMap.lidar_scan_xy(
                distances, angles, x_odom, y_odom, theta_odom
            )
            input_binary_maps = input_gridMap.discretize(distances_x, distances_y)

            # ---- 添加 channel 维 + 前向 + 损失 ----
            input_binary_maps = input_binary_maps.unsqueeze(2)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)

            prediction, kl_loss = model(input_binary_maps)
            ce_loss = criterion(prediction, mask_binary_maps[:, 0]).div(batch_size)
            loss = ce_loss + BETA * kl_loss

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                ce_loss = ce_loss.mean()
                kl_loss = kl_loss.mean()

            running_loss += loss.item()
            kl_avg_loss += kl_loss.item()
            ce_avg_loss += ce_loss.item()

    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter

    return val_loss, val_kl_loss, val_ce_loss


# =============================================================================
# 主函数
# =============================================================================

def main(argv):
    """
    训练入口。

    参数:
        argv[0]: model_path — 模型保存路径（也用于断点续训加载）
        argv[1]: train_dir — 训练数据目录
        argv[2]: val_dir   — 验证数据目录

    流程:
        1. 加载训练/验证数据
        2. 初始化模型、优化器、损失函数
        3. 如果已有模型→加载继续训练
        4. 训练循环 (50 epochs)
        5. 每 epoch: train → validate → TensorBoard 记录
        6. 每 10 epoch: 保存 checkpoint
    """
    # ---- 参数检查 ----
    if len(argv) != NUM_ARGS:
        print("usage: python train.py [MDL_PATH] [TRAIN_PATH] [VAL_PATH]")
        exit(-1)

    mdl_path = argv[0]   # 模型保存路径
    pTrain = argv[1]     # 训练数据目录
    pDev = argv[2]       # 验证数据目录

    # 创建输出目录
    odir = os.path.dirname(mdl_path)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # 1. 加载数据
    # =====================================================================
    print('...Start reading data...')

    # 训练集
    train_dataset = VaeTestDataset(pTrain, 'train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=4,            # 4 个后台线程加载数据
        shuffle=True,             # ★ 训练时打乱数据
        drop_last=True,           # 丢弃不完整的最后一批
        pin_memory=True           # 加速 CPU→GPU 数据传输
    )

    # 验证集
    dev_dataset = VaeTestDataset(pDev, 'val')
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,              # 验证也打乱（不影响指标，因为损失是平均的）
        drop_last=True,
        pin_memory=True
    )

    # =====================================================================
    # 2. 初始化模型
    # =====================================================================
    model = scope(
        input_channels=NUM_INPUT_CHANNELS,    # 1
        latent_dim=NUM_LATENT_DIM,            # 512
        output_channels=NUM_OUTPUT_CHANNELS   # 1
    )
    model.to(device)

    # =====================================================================
    # 3. 优化器 & 损失函数
    # =====================================================================
    # Adam 优化器参数
    opt_params = {
        LEARNING_RATE: 0.001,           # 初始学习率
        BETAS: (.9, 0.999),            # Adam 动量参数
        EPS: 1e-08,                     # 数值稳定性
        WEIGHT_DECAY: .001              # L2 正则化（防止过拟合）
    }

    # ★ BCE 损失: 二值交叉熵
    # reduction='sum': 对所有元素求和（最后手动除以 batch_size）
    # 为什么 sum 而非 mean？保留多 GPU 时的灵活性
    criterion = nn.BCELoss(reduction='sum')
    criterion.to(device)

    optimizer = Adam(model.parameters(), **opt_params)

    # =====================================================================
    # 4. 断点续训
    # =====================================================================
    epochs = NUM_EPOCHS  # 50

    if os.path.exists(mdl_path):
        checkpoint = torch.load(mdl_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load epoch {} success'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models, restart training')

    # =====================================================================
    # 5. 多 GPU 支持（DataParallel）
    # =====================================================================
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # =====================================================================
    # 6. TensorBoard 日志
    # =====================================================================
    writer = SummaryWriter('runs')  # 日志写入 ./runs/

    # =====================================================================
    # 7. 训练循环
    # =====================================================================
    epoch_num = 0
    for epoch in range(start_epoch + 1, epochs):
        # 调整学习率（本项目50轮始终=1e-4）
        adjust_learning_rate(optimizer, epoch)

        # ---- 训练一轮 ----
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss = train(
            model, train_dataloader, train_dataset, device,
            optimizer, criterion, epoch, epochs
        )

        # ---- 验证一轮 ----
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss = validate(
            model, dev_dataloader, dev_dataset, device, criterion
        )

        # ---- TensorBoard 记录 ----
        writer.add_scalar('training loss', train_epoch_loss, epoch)
        writer.add_scalar('training kl loss', train_kl_epoch_loss, epoch)
        writer.add_scalar('training ce loss', train_ce_epoch_loss, epoch)
        writer.add_scalar('validation loss', valid_epoch_loss, epoch)
        writer.add_scalar('validation kl loss', valid_kl_epoch_loss, epoch)
        writer.add_scalar('validation ce loss', valid_ce_epoch_loss, epoch)

        print('Train set: Average loss: {:.4f}'.format(train_epoch_loss))
        print('Validation set: Average loss: {:.4f}'.format(valid_epoch_loss))

        # ---- 每 10 epoch 保存 checkpoint ----
        if epoch % 10 == 0:
            if torch.cuda.device_count() > 1:
                # 多 GPU: 保存 module（去掉 DataParallel 包装）
                state = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
            else:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
            path = './model/model' + str(epoch) + '.pth'
            torch.save(state, path)

        epoch_num = epoch

    # =====================================================================
    # 8. 保存最终模型
    # =====================================================================
    if torch.cuda.device_count() > 1:
        state = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num
        }
    else:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num
        }
    torch.save(state, mdl_path)

    return True


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    main(sys.argv[1:])  # 去掉脚本名，传递参数: [model_path, train_dir, val_dir]
