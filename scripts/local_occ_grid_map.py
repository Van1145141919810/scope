#!/usr/bin/env python
"""
SCOPE 项目 — GPU 加速并行占用栅格建图
======================================
作用：将 LiDAR 扫描数据（1080个距离值）转换为 64×64 二值占用栅格地图。
      整个过程用 PyTorch 向量化实现，在 GPU 上并行处理整个 batch。

核心类: LocalMap —— 管理一个局部栅格地图的创建与更新。

建图流程（6步）：
  1. 预测未来机器人位姿     → origin_pose_prediction()
  2. 历史轨迹坐标变换        → robot_coordinate_transform()
  3. LiDAR 极坐标 → 笛卡尔  → lidar_scan_xy()
  4. 连续坐标 → 离散格点    → discretize()
  5. 更新占用概率 (log-odds) → update() [包含 Bresenham 自由空间]
  6. log-odds → 概率值      → retrieve_p(), calc_MLE()

概率模型: 使用 log-odds 表示，因为 log-odds 可以直接加减（无需 Bayes 公式）:
  log_odds(p)   = ln(p / (1-p))
  p(log_odds)   = 1 - 1/(1 + exp(log_odds))
  P_prior = 0.5  → log_odds = 0  (初始: 未知)
  P_free  = 0.3  → log_odds < 0  (向"空闲"偏移)
  P_occ   = 0.7  → log_odds > 0  (向"占用"偏移)
"""

import torch
from bresenham_torch import bresenhamline  # GPU 版 Bresenham 直线追踪


class LocalMap:
    """局部占用栅格地图 —— 并行处理 batch 中所有样本的所有时间帧。"""

    def __init__(self, X_lim, Y_lim, resolution, p, size=[1, 1], device=None):
        """
        初始化栅格地图。

        参数:
            X_lim:     x 轴范围 [x_min, x_max]，如 [0, 6.4]（米）
            Y_lim:     y 轴范围 [y_min, y_max]，如 [-3.2, 3.2]（米）
            resolution: 栅格分辨率，如 0.1（米/格）
            p:         先验占用概率 P_prior，如 0.5
            size:      [batch_size, time_steps]，默认 [1,1]
            device:    torch 设备 (cpu/cuda)

        地图尺寸: (6.4-0)/0.1 = 64 格, (3.2-(-3.2))/0.1 = 64 格 → 64×64
        """
        self.X_lim = X_lim
        self.Y_lim = Y_lim
        self.resolution = resolution
        self.size = size
        self.device = device

        # 生成 x 和 y 轴的离散坐标序列
        # x: [0, 0.1, 0.2, ..., 6.3]  (64 格)
        # y: [-3.2, -3.1, ..., 3.1]    (64 格)
        x = torch.arange(start=X_lim[0], end=X_lim[1], step=resolution)
        y = torch.arange(start=Y_lim[0], end=Y_lim[1], step=resolution)

        self.x_max = len(x)  # 64
        self.y_max = len(y)  # 64

        # 占用概率矩阵（log-odds 表示）
        # 形状: (batch_size, time_steps, 64, 64)
        # 初始化为先验概率 P_prior 的 log-odds 值
        # P_prior=0.5 → log_odds(0.5) = ln(0.5/0.5) = 0
        self.occ_map = torch.full(
            (self.size[0], self.size[1], self.x_max, self.y_max),
            fill_value=self.log_odds(p)
        )
        if self.device is not None:
            self.occ_map = self.occ_map.to(self.device)

    # =========================================================================
    # 基础工具方法
    # =========================================================================

    def log_odds(self, p):
        """
        概率 → log-odds 转换。

        公式: l(x) = ln(p(x) / (1 - p(x)))

        例: p=0.5 → l=0（完全未知）
            p=0.7 → l≈0.847（偏向占用）
            p=0.3 → l≈-0.847（偏向空闲）
        """
        p = torch.tensor(p)
        return torch.log(p / (1 - p))

    def retrieve_p(self, log_map):
        """
        log-odds → 概率转换（log_odds 的逆操作）。

        公式: p(x) = 1 - 1/(1 + exp(l(x)))
             = sigmoid(log_map)

        这等价于 sigmoid 函数，将 (-∞, +∞) 映射到 (0, 1)。
        """
        prob_map = 1 - 1 / (1 + torch.exp(log_map))
        return prob_map

    # =========================================================================
    # 建图流程方法（按调用顺序排列）
    # =========================================================================

    def lidar_scan_xy(self, distances, angles, x_odom, y_odom, theta_odom):
        """
        ① 极坐标 → 笛卡尔坐标转换。

        LiDAR 测量是极坐标：(距离 r, 角度 θ)。
        需要转换到世界坐标系下的 (x, y)。

        参数:
            distances: (batch, time, 1080) — LiDAR 距离值（米）
            angles:    (1080,) — LiDAR 扫描角度（弧度），范围 -135° ~ 135°
            x_odom:    (batch, time) — 机器人 x 坐标
            y_odom:    (batch, time) — 机器人 y 坐标
            theta_odom:(batch, time) — 机器人朝向角（弧度）

        返回:
            distances_x: (batch, time, 1080) — 每个激光点的世界 x 坐标
            distances_y: (batch, time, 1080) — 每个激光点的世界 y 坐标

        公式:
            x_world = x_robot + r * cos(θ_laser + θ_robot)
            y_world = y_robot + r * sin(θ_laser + θ_robot)
        """
        # 扩展维度以匹配 distances 的形状 (batch, time, 1080)
        angles = angles.expand(distances.size(0), distances.size(1), distances.size(2))
        x_odom = x_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))
        y_odom = y_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))
        theta_odom = theta_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))

        # ★ 极坐标 → 笛卡尔坐标
        # cos(θ_laser + θ_robot)：激光角度 + 机器人朝向 = 世界角度
        distances_x = x_odom + distances * torch.cos(angles + theta_odom)
        distances_y = y_odom + distances * torch.sin(angles + theta_odom)

        return distances_x, distances_y

    def is_valid(self, x_r, y_c):
        """
        检查格点坐标是否在地图范围内。

        参数:
            x_r: 格点 x 索引
            y_c: 格点 y 索引

        返回:
            flag_v: bool 张量，(0,0) ≤ 索引 < (64,64) 且 ≥ 0
        """
        flag_v = (x_r < self.x_max) & (y_c < self.y_max) & (x_r >= 0) & (y_c >= 0)
        return flag_v

    def discretize(self, x, y):
        """
        ② 连续世界坐标 → 离散栅格索引 → 二值栅格地图。

        关键操作:
          1. 物理坐标 (米) → 栅格索引 (格号)
          2. 过滤超出地图范围的激光点
          3. 在对应位置标记为 1（有障碍物）

        参数:
            x: (batch, time, 1080) — 激光点 x 坐标
            y: (batch, time, 1080) — 激光点 y 坐标

        返回:
            binary_map: (batch, time, 64, 64) — 二值占用图（0=空闲, 1=占用）
        """
        # 物理坐标 → 栅格索引
        # 例: x=3.25m, X_lim[0]=0, resolution=0.1
        #     → (3.25-0)/0.1 = 32.5 → floor(32.5) = 格子 32
        x_r = torch.floor((x - self.X_lim[0]) / self.resolution).to(int)
        y_c = torch.floor((y - self.Y_lim[0]) / self.resolution).to(int)

        # 过滤超出地图范围的激光点
        flag_v = self.is_valid(x_r, y_c)           # 找出合法位置
        idx_v = torch.nonzero(flag_v)               # 合法位置的下标

        # 提取合法的格点索引
        x_rv = x_r[idx_v[:, 0], idx_v[:, 1], idx_v[:, 2]]
        y_cv = y_c[idx_v[:, 0], idx_v[:, 1], idx_v[:, 2]]

        # 创建全零的二值图
        binary_map = torch.zeros(self.size[0], self.size[1], self.x_max, self.y_max)
        if self.device is not None:
            binary_map = binary_map.to(self.device)

        # ★ 在激光点命中位置标记为 1（占用）
        binary_map[idx_v[:, 0], idx_v[:, 1], x_rv, y_cv] = 1

        return binary_map

    def update(self, x0, y0, x, y, p_free, p_occ):
        """
        ③ 更新占用栅格概率 —— 同时标记障碍物和自由空间。

        对每条激光射线:
          起点（机器人位置）→ 终点（障碍物位置）
          - 终点格子占用概率 ↑ (log_odds += log_odds(p_occ))
          - 路径上所有格子空闲概率 ↑ (log_odds += log_odds(p_free))

        自由空间通过 Bresenham 算法找到（从机器人到障碍物之间的直线格子）。

        参数:
            x0, y0:  机器人位置（每条射线的起点）
            x, y:    激光点位置（每条射线的终点），形状 (batch, time, 1080)
            p_free:  自由空间概率（如 0.3）
            p_occ:   占用概率（如 0.7）
        """
        # 先离散化终点 → 得到障碍物位置
        binary_map = self.discretize(x, y)

        occ_map = binary_map.clone().detach()
        # 遍历 batch 和时间维度
        for i in range(self.size[0]):      # batch
            for j in range(self.size[1]):  # time
                # 找出该 (batch, time) 下所有障碍物的格点坐标
                end = torch.nonzero(binary_map[i, j])  # (N_obstacles, 2) — (x_r, y_c)

                if end.size(0) != 0:  # 有障碍物，需要标记自由空间
                    # 机器人位置 → 栅格索引
                    x0_r = torch.floor((x0[i, j] - self.X_lim[0]) / self.resolution).to(int)
                    y0_c = torch.floor((y0[i, j] - self.Y_lim[0]) / self.resolution).to(int)

                    # 起点：所有射线共用同一个机器人位置
                    start = torch.tensor([x0_r, y0_c]).to(self.device)
                    start = start.unsqueeze(1).expand(end.size(1), end.size(0)).permute(1, 0)

                    # ★ Bresenham 直线追踪：找出起点到各终点之间的所有格子
                    # points: (总点数, 2) — 沿着所有射线的所有自由格子坐标
                    points = bresenhamline(end, start, max_iter=-1)
                    x_r = points[:, 0]
                    y_c = points[:, 1]

                    # 过滤合法格点
                    flag_v = self.is_valid(x_r, y_c)
                    idx_v = torch.nonzero(flag_v)
                    x_rv = x_r[idx_v[:, 0]]
                    y_cv = y_c[idx_v[:, 0]]

                    # 自由空间标记为 -1（区别于占用格=1 和未观测=0）
                    occ_map[i, j, x_rv, y_cv] = -1

        # ★ 更新 log-odds 概率（可直接加减！）
        # 自由空间: occ_map == -1 → 概率向 "空闲" 偏移
        self.occ_map[occ_map == -1] += self.log_odds(p_free)
        # 占用空间: occ_map == 1  → 概率向 "占用" 偏移
        self.occ_map[occ_map == 1] += self.log_odds(p_occ)

    # =========================================================================
    # 推理/后处理方法
    # =========================================================================

    def calc_MLE(self, prob_map, threshold_p_occ):
        """
        ④ 最大似然估计 (MLE) — 概率值 → 二值决策。

        将连续概率图转换为 0/1 决策:
          prob ≥ 0.8 → 1 (占用)
          prob < 0.8 → 0 (空闲/未知)

        参数:
            prob_map: 概率图 (batch, 64, 64)，值域 [0,1]
            threshold_p_occ: 占用阈值，默认 0.8

        返回:
            prob_map: 二值化后的地图
        """
        prob_map[prob_map >= threshold_p_occ] = 1
        prob_map[prob_map < threshold_p_occ] = 0
        return prob_map

    def to_prob_occ_map(self, threshold_p_occ):
        """
        ⑤ log-odds 地图 → 二值占用图（完整转换链）。

        步骤: 时间帧求和 → log-odds → 概率 → MLE 二值化

        返回:
            prob_map: (batch, 64, 64) 二值占用图
        """
        # sum(dim=1): 将所有时间帧的 log-odds 累加
        log_map = torch.sum(self.occ_map, dim=1)

        # log-odds → 概率
        prob_map = self.retrieve_p(log_map)

        # 概率 → 0/1 二值决策
        prob_map = self.calc_MLE(prob_map, threshold_p_occ)

        return prob_map

    # =========================================================================
    # 运动模型方法
    # =========================================================================

    def origin_pose_prediction(self, vel_N, obs_pos_N, T, noise_std=[0, 0, 0]):
        """
        ⑥ 匀速运动模型 — 预测 T 步后机器人的位姿（坐标原点）。

        为什么需要？机器人在运动，未来 T 步的坐标系原点不同。
        需要将历史观测对齐到预测的未来参考系。

        参数:
            vel_N:    当前速度 (batch, 2) — [线速度 v, 角速度 ω]
            obs_pos_N:当前位姿 (batch, 3) — [x, y, θ]
            T:        预测步数（T=1 表示预测 1 步后的位置）
            noise_std:运动噪声标准差 [σx, σy, σθ]，默认 [0,0,0]

        返回:
            pos_origin: (batch, 3) — 预测的 T 步后位姿

        公式（匀速模型，Δt=0.1s）:
            d = v × 0.1 × T              ← 线速度 × 时间
            θ_new = θ_obs + ω × 0.1 × T  ← 角速度 × 时间
            x_new = x_obs + d × cos(θ_obs) + noise_x
            y_new = y_obs + d × sin(θ_obs) + noise_y
        """
        pos_origin = torch.zeros(self.size[0], 3)

        # 运动噪声（训练时可注入不确定性）
        x_noise = torch.randn(self.size[0]) * noise_std[0]
        y_noise = torch.randn(self.size[0]) * noise_std[1]
        th_noise = torch.randn(self.size[0]) * noise_std[2]

        if self.device is not None:
            pos_origin = pos_origin.to(self.device)
            x_noise = x_noise.to(self.device)
            y_noise = y_noise.to(self.device)
            th_noise = th_noise.to(self.device)

        # 匀速运动学
        d = vel_N[:, 0] * 0.1 * T              # 线速度 × 时间步长 × 预测步数
        theta = vel_N[:, 1] * 0.1 * T           # 角速度产生的旋转角
        pos_origin[:, 0] = obs_pos_N[:, 0] + d * torch.cos(obs_pos_N[:, 2]) + x_noise
        pos_origin[:, 1] = obs_pos_N[:, 1] + d * torch.sin(obs_pos_N[:, 2]) + y_noise
        pos_origin[:, 2] = obs_pos_N[:, 2] + theta + th_noise

        return pos_origin

    def robot_coordinate_transform(self, pos, pos_origin):
        """
        ⑦ 坐标变换 — 将历史轨迹变换到预测参考系下。

        为什么需要？预测的是"未来 T 步之后"的占用图，
        需要把过去10帧的观测对齐到那个未来时刻的坐标系。

        变换: 世界系 → 以 pos_origin 为原点的局部系
              先平移 (-x_origin, -y_origin)，再旋转 -θ_origin

        参数:
            pos:       历史位姿序列 (batch, 10, 3) — [x, y, θ]
            pos_origin:目标参考系位姿 (batch, 3)

        返回:
            x_odom:    变换后的 x 坐标 (batch, 10)
            y_odom:    变换后的 y 坐标 (batch, 10)
            theta_odom:变换后的朝向角 (batch, 10)

        变换公式:
            dx = x_hist - x_origin
            dy = y_hist - y_origin
            x_local =  cos(θ_origin) * dx + sin(θ_origin) * dy
            y_local = -sin(θ_origin) * dx + cos(θ_origin) * dy
            θ_local = θ_hist - θ_origin
        """
        # 扩展 pos_origin 维度以匹配 pos 的 (batch, 10, 3)
        pos_origin = pos_origin.unsqueeze(2).expand(
            pos.size(0), pos.size(2), pos.size(1)
        ).permute(0, 2, 1)

        # 位移分量
        dx = pos[:, :, 0] - pos_origin[:, :, 0]
        dy = pos[:, :, 1] - pos_origin[:, :, 1]
        th = pos_origin[:, :, 2]

        # ★ 旋转矩阵作用于每帧的位移
        # [cos(th)   sin(th)] [dx]
        # [-sin(th)  cos(th)] [dy]
        x_odom = torch.cos(th) * dx + torch.sin(th) * dy
        y_odom = torch.sin(-th) * dx + torch.cos(th) * dy
        theta_odom = pos[:, :, 2] - th

        return x_odom, y_odom, theta_odom
