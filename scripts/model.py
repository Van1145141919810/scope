#!/usr/bin/env python
"""
SCOPE 项目 — 核心模型架构 + 数据集类
======================================
这是整个项目最核心的文件，包含三大部分：
  第一部分：全局常量 + 工具函数
  第二部分：VaeTestDataset — 数据集加载类
  第三部分：模型架构 — 从底层组件到完整 SCOPE 模型

架构层次（底→顶）:
  Residual → ResidualStack → Encoder / Decoder
  VAE_Encoder = Encoder + (μ, logσ) 输出层
  scope = ConvLSTMCell + VAE_Encoder + 重参数化 + Decoder

数据流:
  LiDAR (20帧×1080点) → VaeTestDataset → LocalMap建图 → 10帧栅格
  → ConvLSTM时序编码 → VAE编码(μ,σ) → 重参数化(z) → Decoder
  → 未来占用图预测 [batch, 1, 64, 64]
"""

# =============================================================================
# 第一部分：导入与全局常量
# =============================================================================

from __future__ import print_function  # Python 2/3 兼容（本项目实际需Py3.7+）
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from convlstm import ConvLSTMCell  # 时序编码组件

import os
import random

# ---- 全局常量 ----
SEED1 = 1337         # 随机种子（保证结果可复现）
NEW_LINE = "\n"      # 文件读取用

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 第一部分（续）：工具函数
# =============================================================================

def set_seed(seed):
    """
    设置随机种子以保证实验可复现性。

    策略: 设置 cuDNN 为确定性模式（禁用自动算法搜索）。
    注释掉的 manual_seed 等是因为仅靠 cuDNN 确定性已足够。
    """
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 使用确定性卷积算法
    torch.backends.cudnn.benchmark = False       # 禁用自动调优（防止选择非确定性算法）
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# 第二部分：数据集类
# =============================================================================

POINTS = 1080     # LiDAR 每帧扫描点数
IMG_SIZE = 64     # 栅格地图尺寸 64×64
SEQ_LEN = 10      # ★ 序列长度：10帧历史 → 预测10帧未来


class VaeTestDataset(torch.utils.data.Dataset):
    """
    SCOPE 数据集加载器。

    数据目录结构:
        <dataset_root>/
        ├── scans/train.txt       ← 每行一个 .npy 文件名
        ├── positions/train.txt   ← 对应位姿文件
        └── velocities/train.txt  ← 对应速度文件

    每次 __getitem__ 返回连续的 (SEQ_LEN×2)=20 帧数据：
      前 10 帧 = 历史（用于构建输入栅格）
      后 10 帧 = 未来（用于构建目标/真值栅格）
    """

    def __init__(self, img_path, file_name):
        """
        初始化数据集 — 读取文件列表。

        参数:
            img_path: 数据集根目录，如 '~/data/OGM-datasets/OGM-Turtlebot2/test'
            file_name: 数据集分割名: 'train', 'val', 或 'test'
                      用于读取 {scans,positions,velocities}/{file_name}.txt
        """
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []

        # 打开三个索引文件
        fp_scan = open(img_path + '/scans/' + file_name + '.txt', 'r')
        fp_pos = open(img_path + '/positions/' + file_name + '.txt', 'r')
        fp_vel = open(img_path + '/velocities/' + file_name + '.txt', 'r')

        # 解析文件名列表（跳过非 .npy 的行）
        for line in fp_scan.read().split(NEW_LINE):
            if '.npy' in line:
                self.scan_file_names.append(img_path + '/scans/' + line)
        for line in fp_pos.read().split(NEW_LINE):
            if '.npy' in line:
                self.pos_file_names.append(img_path + '/positions/' + line)
        for line in fp_vel.read().split(NEW_LINE):
            if '.npy' in line:
                self.vel_file_names.append(img_path + '/velocities/' + line)

        fp_scan.close()
        fp_pos.close()
        fp_vel.close()

        self.length = len(self.scan_file_names)
        print("dataset length: ", self.length)  # 例: 10890 (train), 17000 (test)

    def __len__(self):
        """返回数据集总样本数（= 扫描文件数量）。"""
        return self.length

    def __getitem__(self, idx):
        """
        加载一个样本 —— 连续 20 帧数据。

        参数:
            idx: 样本索引起点

        返回:
            dict: {
                'scan':     (20, 1080) — LiDAR 距离值
                'position': (20, 3)    — [x, y, θ] 位姿
                'velocity': (20, 2)    — [v_linear, v_angular] 速度
            }

        边界处理: 如果 idx+20 超出数据集末尾，回绕到 idx-(20)，
                 确保始终有连续 20 帧可用。
        """
        # 初始化 20 帧的零数组
        scans = np.zeros((SEQ_LEN + SEQ_LEN, POINTS))      # (20, 1080)
        positions = np.zeros((SEQ_LEN + SEQ_LEN, 3))       # (20, 3)
        vels = np.zeros((SEQ_LEN + SEQ_LEN, 2))            # (20, 2)

        # ★ 边界回绕：确保 idx+19 不越界
        if idx + (SEQ_LEN + SEQ_LEN) < self.length:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN + SEQ_LEN)

        # 加载连续 20 帧
        for i in range(SEQ_LEN + SEQ_LEN):
            # 加载 LiDAR 扫描 (1080,)
            scan_name = self.scan_file_names[idx_s + i]
            scan = np.load(scan_name)
            scans[i] = scan

            # 加载位姿 (3,) — [x, y, θ]
            pos_name = self.pos_file_names[idx_s + i]
            pos = np.load(pos_name)
            positions[i] = pos

            # 加载速度 (2,) — [v, ω]
            vel_name = self.vel_file_names[idx_s + i]
            vel = np.load(vel_name)
            vels[i] = vel

        # ---- 数据清洗 ----
        # LiDAR 异常值处理:
        #   NaN → 20m（传感器读数无效，设为最大量程）
        #   inf → 20m（溢出）
        #   30  → 20m（传感器返回的最大值标记）
        # 20m 是传感器的有效最大量程
        scans[np.isnan(scans)] = 20.
        scans[np.isinf(scans)] = 20.
        scans[scans == 30] = 20.

        # 位姿和速度异常值 → 0
        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.
        vels[np.isnan(vels)] = 0.
        vels[np.isinf(vels)] = 0.

        # numpy → PyTorch tensor (FloatTensor = float32)
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        vel_tensor = torch.FloatTensor(vels)

        return {
            'scan': scan_tensor,       # (20, 1080)
            'position': pose_tensor,   # (20, 3)
            'velocity': vel_tensor,    # (20, 2)
        }


# =============================================================================
# 第三部分：模型架构组件
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 残差块
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """
    残差块 —— 输入 + 变换后的输出（跳跃连接）。

    结构（瓶颈型）:
        ReLU → 3×3 Conv(128→64) → BN → ReLU → 1×1 Conv(64→128) → BN
        输出 = 输入 + 上述变换

    为什么用 1×1 卷积？
      3×3 Conv: 提取空间特征
      1×1 Conv: 升降维度，减少参数（128→64→128，而非 128→128→128）
              参数对比: 3×3×(128×128) ≈ 147K vs 3×3×(128×64) + 1×1×(64×128) ≈ 82K
    为什么不用 bias？BN 层的均值/方差会抵消 bias 的效果。
    """

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        """
        参数:
            in_channels:          输入通道数（也是输出通道数，保持维度一致）
            num_hiddens:          主干输出通道数
            num_residual_hiddens: 中间瓶颈层通道数
        """
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),                                    # inplace=True 节省显存
            nn.Conv2d(in_channels=in_channels,                # 3×3 卷积：空间特征提取
                      out_channels=num_residual_hiddens,      # 降维（瓶颈）
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),             # 归一化，稳定训练
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,       # 1×1 卷积：通道变换
                      out_channels=num_hiddens,                # 升维回原通道数
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )

    def forward(self, x):
        """
        ★ 残差连接：F(x) = x + block(x)
        恒等映射 + 学习到的残差：
          - 如果 block(x)=0，至少还有原始输入通过（梯度不会消失）
          - 实际学习的是 block(x) = 目标 - x（即"残差"）
        """
        return x + self._block(x)


class ResidualStack(nn.Module):
    """
    残差堆叠 —— N 个 Residual 块的串联。

    末尾用 ReLU 激活（而非恒等映射），增加非线性。
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        参数:
            num_residual_layers: 堆叠的残差块数量（本项目=2）
        """
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        # 创建 N 个相同的 Residual 块
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(self._num_residual_layers)
        ])

    def forward(self, x):
        """串联所有残差块，最后一层额外加 ReLU。"""
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)  # 最终非线性激活


# ---------------------------------------------------------------------------
# 3.2 编码器
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    空间压缩编码器 —— 将特征图逐步下采样。

    数据流:
        输入: (B, 32, 64, 64)  ← ConvLSTM 隐状态
            ↓ conv_1: 4×4 Conv, stride=2, padding=1
        (B, 64, 32, 32)        ← 尺寸减半，通道翻倍
            ↓ conv_2: 4×4 Conv, stride=2, padding=1
        (B, 128, 16, 16)       ← 尺寸再减半，通道再翻倍
            ↓ ResidualStack × 2
        (B, 128, 16, 16)       ← 保持尺寸，增强特征

    为什么用 stride=2 下采样（而非 pooling）？
      卷积自带可学习参数，比固定的 max/avg pooling 更能保留任务相关信息。
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        参数:
            in_channels: 32 — ConvLSTM 隐状态通道数
            num_hiddens: 128 — 编码器最终通道数
        """
        super(Encoder, self).__init__()

        # 第一层下采样: 64×64 → 32×32, 通道: 32 → 64
        self._conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels,        # 32
                      out_channels=num_hiddens // 2,   # 64
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU(True)
        ])

        # 第二层下采样: 32×32 → 16×16, 通道: 64 → 128
        self._conv_2 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_hiddens // 2,   # 64
                      out_channels=num_hiddens,        # 128
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens)
            # 注意：此处无 ReLU！（保持线性输出，让 ResidualStack 接管非线性）
        ])

        # 深度特征处理: 两个 Residual 块（不改变尺寸）
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,            # 128
            num_hiddens=num_hiddens,            # 128
            num_residual_layers=num_residual_layers,  # 2
            num_residual_hiddens=num_residual_hiddens  # 64
        )

    def forward(self, inputs):
        """编码器前向传播：两次下采样 + 残差处理。"""
        x = self._conv_1(inputs)        # (B,32,64,64) → (B,64,32,32)
        x = self._conv_2(x)             # (B,64,32,32) → (B,128,16,16)
        x = self._residual_stack(x)     # (B,128,16,16) → (B,128,16,16)
        return x


# ---------------------------------------------------------------------------
# 3.3 解码器
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    空间恢复解码器 —— 从 latent 表示重建占用栅格图。

    数据流（Encoder 的镜像）:
        输入: (B, 128, 16, 16)  ← latent 表示
            ↓ ResidualStack × 2
            ↓ ConvTranspose2d, stride=2
        (B, 64, 32, 32)
            ↓ ConvTranspose2d, stride=2 → 3×3 Conv → Sigmoid
        (B, 1, 64, 64)          ← 输出占用概率图

    为什么用 ConvTranspose2d（转置卷积）上采样？
      与 Encoder 的下采样对称，卷积核可学习，比固定插值更灵活。
    """

    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        参数:
            out_channels: 1 — 输出占用概率图（单通道）
            num_hiddens:  128 — 解码器输入通道数
        """
        super(Decoder, self).__init__()

        # 深度特征处理（与 Encoder 对称）
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )

        # 第一次上采样: 16×16 → 32×32, 通道: 128 → 64
        self._conv_trans_2 = nn.Sequential(*[
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=num_hiddens,      # 128
                               out_channels=num_hiddens // 2,  # 64
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU(True)
        ])

        # 第二次上采样: 32×32 → 64×64, 通道: 64 → 1
        self._conv_trans_1 = nn.Sequential(*[
            # 上采样 + 特征细化
            nn.ConvTranspose2d(in_channels=num_hiddens // 2,  # 64
                               out_channels=num_hiddens // 2,  # 64（先保持）
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU(True),
            # 细化卷积 + 输出投影
            nn.Conv2d(in_channels=num_hiddens // 2,            # 64
                      out_channels=out_channels,                # 1
                      kernel_size=3, stride=1, padding=1),
            # ★ Sigmoid: 将输出压缩到 (0,1)，表示占用概率
            nn.Sigmoid()
        ])

    def forward(self, inputs):
        """解码器前向传播：残差处理 + 两次上采样 → 概率图。"""
        x = self._residual_stack(inputs)   # (B,128,16,16) → (B,128,16,16)
        x = self._conv_trans_2(x)          # (B,128,16,16) → (B,64,32,32)
        x = self._conv_trans_1(x)          # (B,64,32,32)  → (B,1,64,64)
        return x


# ---------------------------------------------------------------------------
# 3.4 VAE 编码器
# ---------------------------------------------------------------------------

class VAE_Encoder(nn.Module):
    """
    VAE 编码器 —— 输出高斯分布的参数 (μ, log σ)。

    与普通编码器的区别：输出不是单个确定值，而是概率分布的参数。

    数据流:
        ConvLSTM 隐状态 (B, 32, 64, 64)
            ↓ Encoder（2层下采样 + Residual）
        (B, 128, 16, 16)
            ↓ 两路并行的 1×1 Conv
        μ (B, 2, 16, 16)      log σ (B, 2, 16, 16)

    为什么输出 log σ 而不是 σ？
      σ > 0 是约束条件。log σ 可以取任意实数值（无约束），
      通过 exp(log σ) 恢复 σ 时自动保证正定性。
      且 log 能放大微小变化 → 梯度更稳定。
    """

    def __init__(self, input_channel):
        """
        参数:
            input_channel: 32 — ConvLSTM 隐状态通道数
        """
        super(VAE_Encoder, self).__init__()
        self.input_channels = input_channel

        # 超参数
        num_hiddens = 128           # Encoder 内部通道数
        num_residual_hiddens = 64   # 瓶颈通道数
        num_residual_layers = 2     # 残差块数量
        embedding_dim = 2           # latent 空间的通道数

        # 核心编码器: 下采样 + 残差处理
        self._encoder = Encoder(
            input_channel,              # 32
            num_hiddens,                # 128
            num_residual_layers,        # 2
            num_residual_hiddens        # 64
        )

        # ★ 两路并行的 1×1 卷积 — 各输出 2 通道
        # 1×1 卷积的作用: 逐点线性变换，改变通道数但不改变空间尺寸
        # μ 分支: 学习位置的"最可能" latent 值
        self._encoder_z_mu = nn.Conv2d(
            in_channels=num_hiddens,     # 128
            out_channels=embedding_dim,  # 2
            kernel_size=1, stride=1
        )
        # log σ 分支: 学习该位置的不确定性（标准差的对数）
        self._encoder_z_log_sd = nn.Conv2d(
            in_channels=num_hiddens,     # 128
            out_channels=embedding_dim,  # 2
            kernel_size=1, stride=1
        )

    def forward(self, x):
        """
        VAE 编码器前向传播。

        输入:
            x: ConvLSTM 输出隐状态

        输出:
            z_mu:    均值 μ，形状 (B×2×16×16 展开后)
            z_log_sd: 对数标准差 log σ
        """
        # 确保输入形状为 (B, C=32, 64, 64)
        x = x.reshape(-1, self.input_channels, IMG_SIZE, IMG_SIZE)

        # Encoder 编码: (B,32,64,64) → (B,128,16,16)
        encoder_out = self._encoder(x)

        # 两路 1×1 Conv: (B,128,16,16) → (B,2,16,16)
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)

        return z_mu, z_log_sd


# ---------------------------------------------------------------------------
# 3.5 完整 SCOPE 模型 ★★★
# ---------------------------------------------------------------------------

class scope(nn.Module):
    """
    SCOPE: Stochastic Cartographic Occupancy Prediction Engine
    ==========================================================
    核心架构: ConvLSTM 时序编码 + VAE 概率编码 + 重参数化 + 解码器

    模型结构:
        (1) ConvLSTMCell(in=1→hidden=32)
            — 逐帧处理 10 帧历史栅格，输出时序编码特征
        (2) VAE_Encoder(in=32→latent μ/σ)
            — 将时序特征编码为高斯分布参数
        (3) 重参数化: z = μ + ε·σ
            — 从高斯分布采样（可导）
        (4) Decoder(z→out=1)
            — 从 latent 向量解码为未来占用概率图

    通道变化: 1→32→128→2→128→1
    空间变化: 64→64→16→16→16→64
    """

    def __init__(self, input_channels, latent_dim, output_channels):
        """
        参数:
            input_channels:  1  — 输入栅格通道数（单通道二值图）
            latent_dim:      512 — VAE 隐变量总维度 (=2×16×16)
            output_channels: 1  — 输出占用概率图通道数
        """
        super(scope, self).__init__()

        self.input_channels = input_channels    # 1
        self.latent_dim = latent_dim            # 512
        self.output_channels = output_channels  # 1

        # latent 空间的空间尺寸: sqrt(latent_dim / 2) = sqrt(256) = 16
        # 即 2 通道 × 16×16 = 512 维
        self.z_w = int(np.sqrt(latent_dim // 2))

        # 模型超参数
        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        embedding_dim = 2  # VAE 的 μ/σ 各占 2 通道

        # ---- (1) ConvLSTM 时序编码器 ----
        # 输入: 1 通道栅格图 (64×64)
        # 隐状态: 32 通道 (64×64)
        # 卷积核: 3×3 (same padding 保持空间尺寸)
        self._convlstm = ConvLSTMCell(
            input_dim=self.input_channels,   # 1
            hidden_dim=num_hiddens // 4,     # 32 ← 为什么不直接用128？ConvLSTM内部状态×4=128，与后续Encoder对齐
            kernel_size=(3, 3),
            bias=True
        )

        # ---- (2) VAE 编码器 ----
        # 输入: ConvLSTM 输出 (32 通道)
        # 输出: μ 和 log σ (各 2 通道)
        self._encoder = VAE_Encoder(num_hiddens // 4)  # 32

        # ---- (3) 解码器输入投影 ----
        # 将 latent z (2 通道, 16×16) 投影回 (128 通道, 16×16)
        # 1×1 转置卷积: 将 2 通道提升到 128 通道
        self._decoder_z_mu = nn.ConvTranspose2d(
            in_channels=embedding_dim,   # 2
            out_channels=num_hiddens,    # 128
            kernel_size=1, stride=1      # 不改变空间尺寸，仅改变通道数
        )

        # ---- (4) 解码器 ----
        # 输入: (128, 16, 16)
        # 输出: (1, 64, 64) 占用概率图
        self._decoder = Decoder(
            self.output_channels,       # 1
            num_hiddens,                # 128
            num_residual_layers,        # 2
            num_residual_hiddens        # 64
        )

    # =========================================================================
    # 重参数化技巧 (Reparameterization Trick)
    # =========================================================================

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        ★ VAE 的核心技术：重参数化采样 + KL 散度计算。

        问题: 直接从 N(μ, σ²) 采样 z 不可导（无法反向传播梯度）。
        解法: z = μ + ε × σ, 其中 ε ~ N(0, I)
              μ 和 σ 可导，ε 是独立的随机噪声（不需要梯度）。

        KL 散度:
          KL(q(z|x) || p(z)) = E_q[log q(z|x) - log p(z)]
          q(z|x) = N(z | μ, σ²)    — 后验（编码器输出）
          p(z)   = N(z | 0, I)     — 先验（标准正态分布）

          KL 的作用: 正则化 latent 空间，迫使后验趋近先验。
                    这使得 latent 空间紧凑、连续、光滑 → 可插值生成。

        参数:
            z_mu:     均值 μ，形状 (B, 2, 16, 16) 或展开后 (B×2×16×16)
            z_log_sd: 对数标准差 log σ

        返回:
            z:       采样后的 latent 向量，形状 (B, latent_dim=512, 1)
            kl_loss:  KL 散度损失（标量）
        """
        # 展平为 (batch, latent_dim=512, 1)
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)

        # ---- 定义分布 ----
        # 先验 p(z) = N(0, I): 标准正态分布
        pz = torch.distributions.Normal(
            loc=torch.zeros_like(z_mu),    # 均值 = 0
            scale=torch.ones_like(z_log_sd) # 标准差 = 1
        )

        # 后验 q(z|x) = N(μ, σ²): 编码器输出的分布
        # ★ exp(log σ) = σ（确保标准差为正）
        qz_x = torch.distributions.Normal(
            loc=z_mu,
            scale=torch.exp(z_log_sd)
        )

        # ---- ★ 重参数化采样 ----
        # r_sample() 使用重参数化技巧: z = μ + ε × σ
        # rsample() vs sample(): rsample 保留了 μ/σ 的梯度通路
        z = qz_x.rsample()

        # ---- KL 散度 (Monte Carlo 估计) ----
        # KL = E_q[log q(z|x) - log p(z)]
        # log_prob(z): 计算给定分布下 z 的对数概率密度
        # .sum(dim=1): 对 latent 维度求和（每个维度独立）
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        # 负号: 论文中 KL(q||p)，但此处计算的是 p.log_prob - q.log_prob
        # 等价于 -(q.log_prob - p.log_prob)，即 KL 的标准定义
        kl_loss = -kl_divergence.mean()  # batch 平均

        return z, kl_loss

    # =========================================================================
    # 完整前向传播
    # =========================================================================

    def forward(self, x):
        """
        完整前向传播 —— 从栅格序列到未来占用图预测。

        参数:
            x: 输入栅格序列，需要能 reshape 为 (B, 10, 1, 64, 64)

        返回:
            prediction: 预测的占用概率图 (B, 1, 64, 64)
            kl_loss:    VAE 的 KL 散度损失（训练时用）

        处理流程:
            (1) Reshape → 10 帧序列
            (2) ConvLSTM 逐帧编码 → 隐状态 h (编码了时序信息)
            (3) VAE_Encoder → μ, log σ
            (4) 重参数化采样 → latent z + KL 损失
            (5) Decoder → 占用概率图
        """
        # ---- (1) 标准化输入形状: (B, 10, 1, 64, 64) ----
        x = x.reshape(-1, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)
        b, seq_len, c, h, w = x.size()

        # ---- (2) ConvLSTM 时序编码 ----
        # 初始化隐状态和细胞状态为零
        h_enc, enc_state = self._convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )

        # 逐帧处理 10 帧历史栅格
        # h_enc 随时间步更新，累积整合时序信息
        # 最终 h_enc 编码了全部 10 帧的运动和占用信息
        for t in range(seq_len):
            x_in = x[:, t]                              # 取第 t 帧 (B, 1, 64, 64)
            h_enc, enc_state = self._convlstm(
                input_tensor=x_in,
                cur_state=[h_enc, enc_state]             # 状态在帧间传递
            )

        # ---- (3) VAE 编码 ----
        # 将 ConvLSTM 最终隐状态编码为分布参数
        enc_in = h_enc                                   # (B, 32, 64, 64)
        z_mu, z_log_sd = self._encoder(enc_in)           # 各 (B, 2, 16, 16)

        # ---- (4) 重参数化 ----
        # 从 N(μ, σ²) 中采样，同时计算 KL 散度
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)

        # ---- (5) 解码 ----
        # 将 latent 向量 reshape 为空间形式: (B, 2, 16, 16)
        z = z.reshape(-1, 2, self.z_w, self.z_w)

        # 通道投影: (B, 2, 16, 16) → (B, 128, 16, 16)
        x_d = self._decoder_z_mu(z)

        # 空间恢复: (B, 128, 16, 16) → (B, 1, 64, 64)
        prediction = self._decoder(x_d)

        return prediction, kl_loss


# =============================================================================
# 文件结束
# =============================================================================
