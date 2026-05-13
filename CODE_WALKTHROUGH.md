# SCOPE 代码逐行解析

> 用于组会讲解与复习。所有核心代码在 `scripts/` 目录下。
> 顺序：bresenham_torch → convlstm → local_occ_grid_map → model → train → decode_demo

---

## 项目结构

```
scope/
├── README.md                 # 论文说明 + 使用指南
├── environment.yaml           # mamba 环境定义（Python 3.9 + PyTorch 2.7）
├── LICENSE                    # CC BY 4.0
├── quick_demo.py              # 快速 demo 脚本（3 样本，~25 秒）
├── test_env.py                # 环境验证脚本
│
├── run_train.sh               # 训练入口 → 调用 scripts/train.py
├── run_eval_demo.sh           # 推理入口 → 调用 scripts/decode_demo.py
│
├── model/                     # 训练好的模型权重
│   └── scope_model.pth        # 预训练权重 (epoch 40, 730K 参数)
│
├── output/                    # 推理输出：真值 vs 预测对比图
│   ├── quick_mask0.png        # 未来 10 帧真实占用栅格
│   └── quick_pred0.png        # 模型预测的 10 帧
│
├── demo/                      # 论文原版对比 GIF
│   ├── 1.OGM-Turtlebot2_5th_OGM_Prediction_Demo.gif
│   ├── 2.OGM-Jackal_5th_OGM_Prediction_Demo.gif
│   └── 3.OGM-Spot_5th_OGM_Prediction_Demo.gif
│
└── scripts/                   # ★ 所有核心代码
    ├── bresenham_torch.py     # ① GPU 版 Bresenham 直线算法（最底层工具）
    ├── convlstm.py            # ② ConvLSTM 单元（时序编码组件）
    ├── local_occ_grid_map.py  # ③ LiDAR → 占用栅格地图（数据预处理）
    ├── model.py               # ④ 模型架构 + 数据集类（核心）
    ├── train.py               # ⑤ 训练循环（β-VAE 损失）
    └── decode_demo.py         # ⑥ 推理 + 多步预测 + 可视化
```

### 依赖关系（底→顶）

```
bresenham_torch  ←  local_occ_grid_map  ←  train.py
                                       ←  decode_demo.py
convlstm         ←  model.py           ←  train.py
                                       ←  decode_demo.py
```

---

## ① `bresenham_torch.py` — GPU 版 Bresenham 直线追踪

### 为什么需要？

LiDAR 发射激光击中障碍物。从机器人到障碍物的直线上：
- **经过的每个格子** = 空闲 (free)
- **终点格子** = 占用 (occupied)

Bresenham 算法任务：**给定起点终点，找出直线上经过的所有格子**。原版 1962 年串行算法，这里 PyTorch 重写，GPU 并行处理所有射线。

### 3 个函数调用链

```
bresenhamline()              ← 入口：接收起点、终点
    ↓
_bresenhamlines()            ← 核心：批量并行追踪
    ↓
_bresenhamline_nslope()      ← 工具：归一化斜率
```

### 核心逻辑

**`_bresenhamline_nslope`** (line 9)：取每条射线主轴（变化最大维度），归一化斜率使主轴每步走 1 格，次轴按比例。

**`_bresenhamlines`** (line 32)：核心行 `bline = start + nslope × [1,2,3,...]`——沿斜率逐步累加，`torch.round` 取整到格点。

**图解** start=(0,0)→end=(3,2)：
```
步数  累积点        取整     含义
t=0   (0, 0)       (0, 0)   起点
t=1   (1, 0.667)   (1, 1)   自由格
t=2   (2, 1.333)   (2, 1)   自由格
t=3   (3, 2)       (3, 2)   终点=障碍物
```

直线经过了 (0,0)→(1,1)→(2,1)→(3,2) 四个格子。

---

## ② `convlstm.py` — ConvLSTM 时序编码

### LSTM vs ConvLSTM

普通 LSTM 用全连接处理 1D 向量，丢失空间结构。ConvLSTM 把矩阵乘法换为 **卷积**：

```
LSTM:  x(t)、h(t-1) 拼接 → 全连接×4 → 4个门(标量)
ConvLSTM: x[1×64×64]、h[32×64×64] 沿channel拼接[33×64×64] → 3×3 Conv → 4×32×64×64
```

卷积保留了每个空间位置独立的门控。

### 核心：`ConvLSTMCell.forward` (line 42-57)

```python
h_cur, c_cur = cur_state                          # c=记忆细胞, h=隐状态
combined = torch.cat([input_tensor, h_cur], dim=1) # 沿channel拼接
combined_conv = self.conv(combined)                 # 一次卷积算4个门
cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, hidden_dim, dim=1)

i = sigmoid(cc_i)   # 输入门：哪些新信息写入记忆
f = sigmoid(cc_f)   # 遗忘门：哪些旧记忆丢弃
o = sigmoid(cc_o)   # 输出门：哪些输出给下一层
g = tanh(cc_g)      # 候选记忆：新信息的具体内容

c_next = f * c_cur + i * g      # 更新：遗忘旧的 + 写入新的
h_next = o * tanh(c_next)        # 输出：过滤后的记忆
```

**门控直观理解**：i=1 "这个障碍物是新的，记住" | f=0 "这个人走远了，忘掉" | o 控制暴露什么。

### 在本项目中的作用

10 帧历史栅格依次送入 ConvLSTMCell，最终隐状态 `h` 编码了 10 帧的运动信息：
```
帧1→ConvLSTMCell→帧2→...→帧10→ConvLSTMCell→h(编码了时序运动特征)
```

---

## ③ `local_occ_grid_map.py` — GPU 并行占用栅格建图

### 为什么需要？

LiDAR 返回 1080 个距离值，不是栅格地图。传统建图用 `for` 循环逐个处理，极慢。此文件用纯 PyTorch 实现 GPU 批量并行建图。

### 核心类 `LocalMap`

**`__init__`** (line 32)：64×64 栅格，0.1m 分辨率，覆盖 6.4m×6.4m。概率用 **log-odds** 表示（可直接加减）。初始化为先验概率 50%。

**① `lidar_scan_xy`** (line 78)：极坐标→笛卡尔坐标
```python
x = x_odom + distances * cos(angles + theta)
y = y_odom + distances * sin(angles + theta)
```

**② `discretize`** (line 102)：连续坐标→离散格点，碰撞格=1，其余=0。

**③ `update`** (line 124)：调用 Bresenham 找自由格 → 更新 log-odds：
```python
self.occ_map[自由格] += log_odds(p_free)   # 向"free"偏移
self.occ_map[占用格] += log_odds(p_occ)    # 向"occupied"偏移
```

**④ `origin_pose_prediction`** (line 177)：匀速模型预测 T 步后机器人位姿：
```python
d = v_linear * 0.1 * T    # 线速度×时间=位移
theta = v_angular * 0.1 * T
new_x = obs_x + d * cos(heading)
new_y = obs_y + d * sin(heading)
```

**⑤ `robot_coordinate_transform`** (line 200)：把历史轨迹对齐到预测参考系（绕原点旋转）。

### 完整建图流程

```
1. 拿到10帧LiDAR数据(1080距离值/帧)
2. origin_pose_prediction→预测未来机器人位姿
3. robot_coordinate_transform→过去轨迹对齐到预测系
4. lidar_scan_xy→极坐标→笛卡尔
5. discretize→连续→0/1二值栅格
结果：10帧 64×64 二值占用图
```

---

## ④ `model.py` — 核心模型架构 + 数据集

这是整个项目最重要的文件（411 行），包含三大部分：**全局常量** → **数据集类** → **模型架构**。

### 4.1 全局常量 (line 30-32, 70-72)

```python
SEED1 = 1337          # 随机种子
POINTS = 1080         # LiDAR 扫描点数（每帧）
IMG_SIZE = 64         # 栅格地图尺寸 64×64
SEQ_LEN = 10          # 关键！10 帧历史 → 预测 10 帧未来
```

`set_seed()` 设置 cuDNN 为确定性模式，保证结果可复现。

### 4.2 数据集类 `VaeTestDataset` (line 73-156)

**数据存储格式**：OGM-Datasets 按目录组织
```
OGM-Turtlebot2/test/
├── scans/train.txt  →  每行列出一个 .npy 文件名
├── positions/train.txt
├── velocities/train.txt
```

**`__init__`** (line 74-98)：读取三个 `.txt` 文件，构建 `.npy` 文件路径列表。

**`__getitem__`** (line 106-153)：核心数据加载逻辑
```python
scans = np.zeros((SEQ_LEN + SEQ_LEN, POINTS))  # (20帧, 1080点)
# 前10帧=历史输入，后10帧=预测目标(mask)
for i in range(SEQ_LEN + SEQ_LEN):  # 连续加载20帧
    scan = np.load(scan_name)
    scans[i] = scan

# 数据清洗：
scans[np.isnan(scans)] = 20.    # NaN→20m（超出范围值）
scans[scans==30] = 20.           # 30m→20m（传感器最大值）

# ★ 关键：返回的 scan tensor 同时包含历史（前10帧）和目标（后10帧）
# 具体怎么切分由 train.py/decode_demo.py 决定，见下文
```

返回 `{'scan': (20,1080), 'position': (20,3), 'velocity': (20,2)}`。

### 4.3 模型组件

#### Residual 块 (line 169-198)

```python
class Residual(nn.Module):
    def forward(self, x):
        return x + self._block(x)  # ★ 残差连接：输入 + 变换
```

本质：3×3 卷积 → BN → ReLU → 1×1 卷积 → BN。**1×1 卷积**用于降低参数量（128→64→128），瓶颈结构。

`ResidualStack` 就是 N 个 Residual 串联 + 最后一层 ReLU。

#### Encoder（空间压缩，line 202-232）

```
输入 x: [batch, 32, 64, 64]   (ConvLSTM 输出的 h)
    ↓ conv_1: 4×4 Conv, stride=2
[batch, 64, 32, 32]
    ↓ conv_2: 4×4 Conv, stride=2
[batch, 128, 16, 16]
    ↓ ResidualStack × 2
[batch, 128, 16, 16]          (空间尺寸不变)
```

两次 stride=2 下采样：64×64 → 32×32 → 16×16。

#### Decoder（空间恢复，line 235-275）

```
输入 z: [batch, 128, 16, 16]
    ↓ ResidualStack × 2
    ↓ ConvTranspose2d stride=2
[batch, 64, 32, 32]
    ↓ ConvTranspose2d stride=2 + Conv 3×3 + Sigmoid
[batch, 1, 64, 64]  ← 输出占用概率图
```

Encoder 的镜像：16×16 → 32×32 → 64×64。最后 **Sigmoid** 输出 (0,1) 概率值。

#### VAE_Encoder（line 277-313）

```
ConvLSTM输出h [batch, 32, 64, 64]
    ↓ Encoder (2层下采样 + Residual)
[batch, 128, 16, 16]
    ↓ 1×1 Conv 分两路
μ [batch, 2, 16, 16]        log σ [batch, 2, 16, 16]
```

**为什么输出 μ 和 log σ？** VAE 学习的是分布而非确定值。编码器输出高斯分布的均值 μ 和对数标准差 log σ，然后在 latent space 采样 z ~ N(μ, σ²)。

### 4.4 完整模型 `scope` (line 316-404)

#### `__init__` — 三组件串联

```
(1) ConvLSTMCell(1→32)      ← 时序编码器
(2) VAE_Encoder(32→latent)  ← VAE 编码
(3) Decoder(128→1)           ← 解码器
```

通道数：1(输入占用图) → 32(ConvLSTM隐层) → 128(VAE编码) → 2(latent μ/σ) → 128(解码) → 1(输出占用图)

Latent dim=512 → 展开为 2×16×16（2 通道 × 16×16 空间），`z_w = sqrt(512/2) = 16`。

#### `vae_reparameterize` (line 349-371) — 重参数化技巧

这是 VAE 最关键的技术：

```python
# p(z) = N(0, 1)   — 先验分布（标准正态）
pz = Normal(loc=0, scale=1)

# q(z|x) = N(μ, σ²) — 后验分布（编码器输出）
qz_x = Normal(loc=z_mu, scale=exp(z_log_sd))

# ★ 重参数化：z = μ + ε·σ,  ε~N(0,1)
z = qz_x.rsample()

# KL 散度（Monte Carlo 估计）：
kl = log(qz_x(z)) - log(pz(z))   # 每个 latent 维度
kl_loss = -kl.mean()             # batch 平均
```

**为什么需要重参数化？** 采样操作不可导（`z = μ + ε·σ` 中 ε 是随机的），但 μ 和 σ 可导。这个技巧让梯度能穿过采样，使 VAE 可端到端训练。

**KL 散度的作用**：让后验分布 q(z|x) 趋近先验 p(z)=N(0,1)。这迫使 Encoder 学到紧凑、连续、光滑的 latent 表示。

#### `forward` (line 373-404) — 完整前向传播

```python
# Step 1: reshape 输入
x = x.reshape(-1, 10, 1, 64, 64)  # (batch, seq, ch, H, W)

# Step 2: ConvLSTM 时序编码
h_enc, c = init_hidden(batch_size=b)      # 初始化 h=0, c=0
for t in range(10):                        # 遍历 10 帧
    h_enc, c = convlstm(x[:, t], [h_enc, c])
# 循环结束后 h_enc 编码了 10 帧的时序运动信息

# Step 3: VAE 编码
z_mu, z_log_sd = self._encoder(h_enc)    # 输出 μ 和 log σ

# Step 4: 重参数化采样
z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)

# Step 5: 解码
z = z.reshape(-1, 2, 16, 16)             # latent→空间形式
x_d = self._decoder_z_mu(z)              # 1×1 转置卷积升维
prediction = self._decoder(x_d)           # 解码→占用概率图
return prediction, kl_loss
```

### 端到端数据流

```
10 帧 LiDAR → LocalMap.discretize →
10 帧 64×64 二值栅格 (batch, 10, 1, 64, 64)
    ↓ [逐帧送入 ConvLSTM]
h [batch, 32, 64, 64] ← 编码了 10 帧运动信息
    ↓ [VAE Encoder: 下采样 + 卷积]
μ [batch, 2, 16, 16], log σ [batch, 2, 16, 16]  → KL 损失
    ↓ [重参数化: z = μ + ε·σ]
z [batch, 512]
    ↓ [reshape + Decoder: 上采样 + Sigmoid]
prediction [batch, 1, 64, 64]  ← 未来占用概率图
```

**损失函数**（在 train.py 中）：`L = BCE(prediction, ground_truth) + 0.01 × KL_divergence`

- BCE：预测的占用图是否接近真实
- KL：latent 分布是否接近标准正态（正则化）

---

## ⑤ `train.py` — 训练循环

训练脚本的核心是 **β-VAE 训练**，β=0.01 控制 KL 正则化强度。

### 全局配置 (line 47-71)

```python
NUM_EPOCHS = 50
BATCH_SIZE = 128
NUM_LATENT_DIM = 512
BETA = 0.01               # KL 散度权重（β-VAE 的 β）
MAP_X_LIMIT = [0, 6.4]    # 地图范围
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1          # 0.1m/格
```

### `main` 函数 (line 326-462)

```python
# 1. 加载数据
train_dataset = VaeTestDataset(pTrain, 'train')
train_dataloader = DataLoader(train_dataset, batch_size=128, ...)

# 2. 初始化模型
model = scope(input_channels=1, latent_dim=512, output_channels=1)
model.to(device)

# 3. 定义优化器和损失
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss(reduction='sum')   # ★ 二值交叉熵
criterion.to(device)

# 4. 断点续训
if os.path.exists(mdl_path):
    checkpoint = torch.load(mdl_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

# 5. 训练循环
for epoch in range(start_epoch+1, epochs):
    train_loss = train(model, dataloader, ...)  # 训练一轮
    val_loss = validate(model, dataloader, ...)  # 验证
    writer.add_scalar('training loss', train_loss, epoch)  # TensorBoard
    if epoch % 10 == 0:
        torch.save(state, f'./model/model{epoch}.pth')  # 每10轮保存
```

### `train` 函数的关键步骤 (line 95-205)

```python
for i, batch in enumerate(dataloader):
    scans = batch['scan'].to(device)          # (128, 20, 1080)
    positions = batch['position'].to(device)  # (128, 20, 3)
    velocities = batch['velocity'].to(device) # (128, 20, 2)

    # === 构建目标 (mask)：未来10帧的真值 ===
    distances = scans[:, 10:20]         # 后10帧 LiDAR
    # ... 建图流程 → mask_binary_maps (128, 10, 1, 64, 64)

    # === 构建输入：过去10帧 ===
    pos_origin = origin_pose_prediction(vel_N, obs_pos_N, T=1)
    # 把过去轨迹变换到"预测的未来参考系"
    x_odom, y_odom = robot_coordinate_transform(pos[0:10], pos_origin)
    distances = scans[:, 0:10]         # 前10帧 LiDAR
    # ... 建图流程 → input_binary_maps (128, 10, 1, 64, 64)

    # === 前向传播 ===
    prediction, kl_loss = model(input_binary_maps)

    # === 损失 ===
    ce_loss = BCELoss(prediction, mask_binary_maps[:, 0]) / batch_size
    # ★ 只和 mask 的第0帧比较！即预测1步后的未来
    loss = ce_loss + 0.01 * kl_loss   # β-VAE 损失
    loss.backward()
    optimizer.step()
```

**重要细节**：训练时只用 `mask_binary_maps[:, 0]`（未来第 1 帧）做监督。但在推理时（decode_demo.py）会做自回归 10 步预测——把预测输出当作下一帧的输入，误差会累积。

---

## ⑥ `decode_demo.py` — 推理与可视化

### 与训练的关键区别

| | train.py | decode_demo.py |
|------|----------|----------------|
| 预测步数 | 1 步（只用 mask[:, 0]） | **10 步自回归** |
| 采样方式 | 1 次前向 | **32 次 Monte Carlo → 取均值** |
| 输入构造 | 每个样本不同参考系 | 逐步更新预测位姿 `T=j+1` |
| 输出 | 损失值 | 可视化对比图 |

### 自回归多步预测逻辑 (line 177-220)

```python
for j in range(10):                      # 预测未来 10 帧
    T = j + 1                            # 预测超前步数
    pos_origin = origin_pose_prediction(vel_N, obs_pos_N, T)
    # ★ T 逐渐增大：1→2→...→10，预测越来越远的未来
    # 把历史轨迹对齐到未来第 T 步的参考系
    x_odom, y_odom = robot_coordinate_transform(pos_hist, pos_origin)
    input_binary = discretize(lidar_scan_xy(...))

    # MC 采样：复制 32 份，独立前向
    inputs_samples = input_binary.repeat(32, 1, 1, 1, 1)

    for t in range(T):                   # 自回归！
        prediction, _ = model(inputs_samples)
        prediction = prediction.reshape(-1, 1, 1, 64, 64)
        inputs_samples = cat([inputs_samples[:, 1:], prediction], dim=1)
        # ★ 扔掉最旧的一帧，预测作为新输入 → 循环

    predictions = prediction.squeeze(1)
    pred_mean = torch.mean(predictions, dim=0)  # 32 次取均值
    prediction_maps[j, 0] = pred_mean
```

**为什么需要 32 次 MC 采样？** VAE 的 latent 空间是概率分布，每次采样结果不同。32 次均值为期望值，减少了单次采样的随机噪声。

### 可视化 (line 222-253)

每处理一个样本，生成两张图：
- `output/mask{i}.jpg`：10 帧真实未来
- `output/pred{i}.jpg`：10 帧预测

### 为什么全量跑太慢？

17000 个测试样本，每个做 10 步自回归 × 32 MC × (1+2+...+10) = 320 次前向/样本 → **~544 万次前向传播**。`quick_demo.py` 限制 3 样本解决。

---

