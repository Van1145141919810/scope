#!/usr/bin/env python
"""
SCOPE 项目 — 推理与可视化脚本
===============================
加载预训练模型，在测试集上执行多步自回归预测，生成真实值 vs 预测值对比图。

与训练的关键区别:
  | 特性       | train.py              | decode_demo.py              |
  |-----------|-----------------------|-----------------------------|
  | 预测步数    | 1 步（只用 mask[:,0]） | ★ 10 步自回归                |
  | 采样方式    | 1 次前向               | ★ 32 次 Monte Carlo → 取均值  |
  | 监督信号    | 需要 ground truth      | 不需要（纯推理）               |
  | 输出       | 损失值                  | mask{i}.jpg + pred{i}.jpg   |

自回归多步预测原理:
  第1步: 输入=真实10帧历史 → 预测帧1
  第2步: 输入=真实9帧 + 预测帧1 → 预测帧2
  ...
  第10步: 输入=真实0帧 + 预测帧1~9 → 预测帧10
  ★ 误差随步数累积（长期预测精度下降）

用法:
  python decode_demo.py <model_path> <test_dir>
  例: python decode_demo.py ../model/scope_model.pth ~/OGM-datasets/OGM-Turtlebot2/test
"""

# ---- 导入 ----
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from tensorboardX import SummaryWriter       # TensorBoard 日志
import matplotlib.pyplot as plt              # 画图
import numpy as np

import torchvision.transforms as transforms
import torchvision
import matplotlib
from torchvision.utils import make_grid     # 将多张图合并为一张网格图
matplotlib.style.use('ggplot')               # ggplot 画图风格

import sys
import os

# 从同目录导入模型和数据加载
from model import *                           # scope模型 + VaeTestDataset + 常量
from local_occ_grid_map import LocalMap       # GPU 并行建图


# =============================================================================
# 全局配置
# =============================================================================

NUM_ARGS = 2                # 命令行参数数量: [model_path, test_dir]
IMG_SIZE = 64               # 栅格图尺寸
SPACE = " "
log_dir = '../model/model.pth'

# 模型常量
NUM_CLASSES = 1
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512
NUM_OUTPUT_CHANNELS = NUM_CLASSES

# 建图参数（必须与训练时一致）
P_prior = 0.5               # 先验占用概率
P_occ = 0.7                 # 占用置信概率（击中）
P_free = 0.3                # 空闲置信概率（穿过）
MAP_X_LIMIT = [0, 6.4]      # x 轴范围（米）
MAP_Y_LIMIT = [-3.2, 3.2]   # y 轴范围（米）
RESOLUTION = 0.1            # 栅格分辨率（米/格）
TRESHOLD_P_OCC = 0.8        # 占用判定阈值

# 设置随机种子
set_seed(SEED1)


# =============================================================================
# 主函数
# =============================================================================

def main(argv):
    """推理主函数。"""

    # ---- 参数检查 ----
    if len(argv) != NUM_ARGS:
        print("usage: python decode_demo.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    mdl_path = argv[0]  # 模型路径，如 '../model/scope_model.pth'
    fImg = argv[1]      # 测试数据目录

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # 1. 加载测试数据
    # =====================================================================
    eval_dataset = VaeTestDataset(fImg, 'test')
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,       # ★ 逐样本处理（每次只处理 1 个样本，用于可视化）
        shuffle=False,      # 不打乱，按顺序处理
        drop_last=True
    )

    # =====================================================================
    # 2. 加载模型
    # =====================================================================
    model = scope(
        input_channels=NUM_INPUT_CHANNELS,    # 1
        latent_dim=NUM_LATENT_DIM,            # 512
        output_channels=NUM_OUTPUT_CHANNELS   # 1
    )
    model.to(device)
    model.eval()  # ★ 切换到评估模式（禁用 BN、Dropout 等训练行为）

    # 定义损失函数（推理时仅参考，不实际用于更新）
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)

    # ★ 加载预训练权重
    checkpoint = torch.load(mdl_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # =====================================================================
    # 3. 推理循环（遍历测试集）
    # =====================================================================
    counter = 0
    num_batches = int(len(eval_dataset) / eval_dataloader.batch_size)

    with torch.no_grad():  # ★ 不计算梯度 → 更快、更省显存
        for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
            counter += 1

            # ---- 提取数据 ----
            scans = batch['scan'].to(device)           # (1, 20, 1080)
            positions = batch['position'].to(device)   # (1, 20, 3) [x, y, θ]
            velocities = batch['velocity'].to(device)  # (1, 20, 2) [v, ω]
            batch_size = scans.size(0)                 # 1

            # =================================================================
            # Step A: 构建目标栅格（mask）—— 未来 10 帧真值（用于对比展示）
            # =================================================================
            mask_gridMap = LocalMap(
                X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT,
                resolution=RESOLUTION, p=P_prior,
                size=[batch_size, SEQ_LEN], device=device
            )

            # 机器人位姿置零（对于 gt mask，使用世界坐标系原点）
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)

            # 后 10 帧 LiDAR（未来帧 = ground truth）
            distances = scans[:, SEQ_LEN:]             # (1, 10, 1080)

            # LiDAR 角度：-135° ~ 135°，共 1080 个角度均匀分布
            angles = torch.linspace(
                -(135 * np.pi / 180), 135 * np.pi / 180,
                distances.shape[-1]                    # 1080
            ).to(device)

            # 极坐标 → 笛卡尔 → 离散化 → 二值栅格
            distances_x, distances_y = mask_gridMap.lidar_scan_xy(
                distances, angles, x_odom, y_odom, theta_odom
            )
            mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
            # 添加 channel 维: (1, 10, 64, 64) → (1, 10, 1, 64, 64)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)

            # =================================================================
            # Step B: ★ 自回归多步预测
            # =================================================================
            # 存储 10 步的预测结果
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)

            for j in range(SEQ_LEN):  # j = 0, 1, ..., 9 → 预测未来第 j+1 帧
                # B1. 创建新的栅格建图器
                input_gridMap = LocalMap(
                    X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT,
                    resolution=RESOLUTION, p=P_prior,
                    size=[batch_size, SEQ_LEN], device=device
                )

                # B2. 预测未来第 T 步的参考系原点
                obs_pos_N = positions[:, SEQ_LEN - 1]  # 当前时刻位姿（第 10 帧）
                vel_N = velocities[:, SEQ_LEN - 1]     # 当前速度
                T = j + 1                                # ★ T=1,2,...,10 逐帧递增
                noise_std = [0, 0, 0]                   # 推理不用噪声

                # 匀速模型预测 T 步后的机器人位姿
                pos_origin = input_gridMap.origin_pose_prediction(
                    vel_N, obs_pos_N, T, noise_std
                )

                # B3. 坐标变换：将历史位姿对齐到预测的参考系
                pos = positions[:, :SEQ_LEN]  # (1, 10, 3)
                x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(
                    pos, pos_origin
                )

                # B4. 取前 10 帧 LiDAR 并建图
                distances = scans[:, :SEQ_LEN]          # (1, 10, 1080)
                distances_x, distances_y = input_gridMap.lidar_scan_xy(
                    distances, angles, x_odom, y_odom, theta_odom
                )
                input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
                input_binary_maps = input_binary_maps.unsqueeze(2)  # (1,10,1,64,64)

                # =============================================================
                # B5. ★ Monte Carlo 采样 + 自回归预测循环
                # =============================================================
                num_samples = 32  # MC 采样次数

                # 复制 32 份：使每个样本有独立的前向传播路径
                # (1,10,1,64,64) → (32,10,1,64,64)
                inputs_samples = input_binary_maps.repeat(num_samples, 1, 1, 1, 1)

                # 自回归：用上一步的预测替换最旧的历史帧
                for t in range(T):
                    # 前向传播（32 个样本同时处理）
                    prediction, kl_loss = model(inputs_samples)
                    # reshape: (32, 1, 64, 64) → (32, 1, 1, 64, 64)
                    prediction = prediction.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)

                    # ★ 自回归核心操作：
                    # 丢弃最旧的一帧 [:, 1:]，将新预测接在末尾
                    # [帧_t, 帧_{t+1}, ..., 帧_{t+9}] → [帧_{t+1}, ..., 帧_{t+9}, 预测]
                    inputs_samples = torch.cat(
                        [inputs_samples[:, 1:], prediction], dim=1
                    )

                # 取最后一帧的预测（即 T 步之后的预测结果）
                predictions = prediction.squeeze(1)  # (32, 1, 64, 64)

                # ★ 32 次预测取均值（消除单次采样的随机性）
                pred_mean = torch.mean(predictions, dim=0, keepdim=True)  # (1, 1, 64, 64)
                prediction_maps[j, 0] = pred_mean.squeeze()

            # =================================================================
            # Step C: 可视化 —— 生成真实值 vs 预测值对比图
            # =================================================================

            # ---- C1. 真实值图（mask）—— 10 帧并排 ----
            fig = plt.figure(figsize=(8, 1))  # 宽8 高1英寸 = 横向长条布局
            for m in range(SEQ_LEN):
                a = fig.add_subplot(1, 10, m + 1)   # 1行10列，第 m+1 个
                mask = mask_binary_maps[0, m]        # 提取第 m 帧
                input_grid = make_grid(mask.detach().cpu())  # (C,H,W)→网格图像
                input_image = input_grid.permute(1, 2, 0)    # (C,H,W)→(H,W,C)
                plt.imshow(input_image)
                plt.xticks([])                                # 隐藏坐标轴
                plt.yticks([])
                fontsize = 8
                input_title = "n=" + str(m + 1)               # n=1, n=2, ..., n=10
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = "./output/mask" + str(i) + ".jpg"
            plt.savefig(input_img_name)  # 保存真值对比图
            plt.close(fig)               # 释放图形内存

            # ---- C2. 预测值图（prediction）—— 10 帧并排 ----
            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):
                a = fig.add_subplot(1, 10, m + 1)
                pred = prediction_maps[m]                     # 提取第 m 步预测
                input_grid = make_grid(pred.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                input_title = "n=" + str(m + 1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = "./output/pred" + str(i) + ".jpg"
            plt.savefig(input_img_name)  # 保存预测对比图
            plt.close(fig)

            # 显示图片（如果有图形界面）
            plt.show()

            print(i)  # 打印当前处理的样本序号

    return True


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    main(sys.argv[1:])  # 去掉脚本名，传递: [model_path, test_dir]
