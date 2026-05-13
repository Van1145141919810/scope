"""
SCOPE 项目 — GPU 版 Bresenham 直线追踪算法
============================================
作用：给定起点和终点，找出直线上经过的所有栅格格子。
用途：在占用栅格建图中，从机器人（起点）到激光点（终点）之间，
      经过的格子标记为"自由"(free)，终点标记为"占用"(occupied)。

原算法：Bresenham's line algorithm (1962)，串行逐个像素绘制。
本实现：用 PyTorch 向量化，GPU 上同时追踪数百条射线。

Modified from: https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
"""

import torch

# 设备选择：优先 GPU，否则 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bresenhamline_nslope(slope):
    """
    归一化斜率向量 —— Bresenham 算法的第一步预处理。

    核心思想：对每条射线，找出变化量最大的维度（主轴），
    让主轴每次移动 1 格，其余维度按比例移动。

    参数:
        slope: 形状 (N, dim) — N 条射线，每条有 dim 维的斜率向量
               slope = end - start，即终点减起点
    返回:
        normalizedslope: 归一化后的斜率，主轴方向 = ±1

    示例:
        start=(0,0), end=(3,2) → slope=(3,2)
        scale = amax(|3|,|2|) = 3
        nslope = (3/3, 2/3) = (1, 0.667)
        意味着：x 轴每步走 1 格，y 轴每步走 0.667 格
    """
    # 沿 dim=1（各维度）取绝对值，再取最大值 → 每条射线的主轴变化量
    # 形状: (N,) → reshape 为 (N, 1) 方便广播
    scale = torch.amax(torch.abs(slope), dim=1).reshape(-1, 1)

    # 处理退化情况：如果斜率为全零（起点=终点），scale=0 会导致除零
    # 将全零行的 scale 设为 1，避免 NaN
    zeroslope = (scale == 0).all(1)          # 找出所有维度变化量都为 0 的射线
    scale[zeroslope] = torch.ones(1, dtype=torch.long).to(device)

    # 归一化：主轴方向 = 1，次轴方向按比例缩放
    normalizedslope = slope / scale
    normalizedslope[zeroslope] = torch.zeros(slope[0].shape).to(device)

    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Bresenham 算法的核心循环 — 批量并行追踪多条射线。

    参数:
        start:   形状 (N, dim) — N 条射线的起点坐标
        end:     形状 (N, dim) — N 条射线的终点坐标
        max_iter: 最大追踪步数。-1 表示自动计算（取所有射线中最长的那条）

    返回:
        bline_points: 形状 (N, max_iter, dim) — 每条射线在每一步的坐标（已取整）

    原理:
        从起点出发，每一步沿着归一化斜率前进，然后用 round() 取整到最近的整数格点。
        这等价于经典 Bresenham 的决策参数方法，但用向量化实现。

    示例:
        start=(0,0), end=(3,2), max_iter=3
        → stepmat = [[1], [2], [3]]
        → bline = (0,0) + (1,0.667)×[[1],[2],[3]]
              = [[1, 0.667], [2, 1.333], [3, 2]]
        → round → [[1,1], [2,1], [3,2]]  ← 直线上经过的 3 个点
    """
    # 自动计算最大步数：取所有射线中终点与起点在各个维度差值的最大绝对值
    # 例如 start=(0,0), end=(3,2)，差值为 (3,2)，amax=3 → 需要追踪 3 步
    if max_iter == -1:
        max_iter = torch.amax(torch.amax(torch.abs(end - start), dim=1))

    npts, dim = start.shape       # npts=射线数量, dim=坐标维度(本项目为2: x和y)
    nslope = _bresenhamline_nslope(end - start)  # 计算归一化斜率 (N, dim)

    # 构建步数矩阵：形状 (max_iter, dim)
    # stepseq = [1, 2, 3, ..., max_iter]
    stepseq = torch.arange(1, max_iter + 1).to(device)

    # 将步数序列复制到每个维度
    # stepmat: (max_iter, dim)，每列都是 [1,2,3,...,max_iter]
    stepmat = stepseq.repeat(dim, 1)  # (dim, max_iter)
    stepmat = stepmat.T               # (max_iter, dim)

    # ★ 核心行 ★
    # start[:, None, :] : (N, 1, dim) — 每条射线的起点
    # nslope[:, None, :] : (N, 1, dim) — 每条射线的归一化斜率
    # stepmat : (max_iter, dim) — 步数 [1,2,3,...]
    # 广播后: (N, max_iter, dim) — 每条射线在每个步数的累积位置
    # 物理意义: 位置 = 起点 + 归一化斜率 × 步数
    bline = start[:, None, :] + nslope[:, None, :] * stepmat

    # 四舍五入到最近的整数格点（离散化到栅格坐标）
    bline_points = torch.round(bline).to(start.dtype)

    return bline_points


def bresenhamline(start, end, max_iter=5):
    """
    Bresenham 直线追踪的对外接口。

    参数:
        start: 形状 (N, dim) — 起点坐标数组
        end:   形状 (1, dim) 或 (N, dim) — 终点坐标
               - 如果是 (1, dim)：所有射线共用一个终点
               - 如果是 (N, dim)：每条射线有自己的终点
        max_iter: 最大追踪步数。-1 = 自动选择最长距离

    返回:
        linevox: 形状 (N*max_iter, dim) — 所有射线经过的所有点（展开为一维）

    示例:
        >>> start = tensor([[3, 1], [0, 0]])    # 2条射线
        >>> end   = tensor([[0, 0], [0, 0]])     # 共用终点(0,0)
        >>> bresenhamline(start, end, max_iter=-1)
        # 返回 2 条射线从各自起点到 (0,0) 沿途的所有格子坐标
    """
    # 调用核心追踪函数，然后展开为 (总点数, dim) 的形状
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])
