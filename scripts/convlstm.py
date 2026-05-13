"""
SCOPE 项目 — ConvLSTM 时序编码器
=================================
作用：对 10 帧栅格地图序列做时序编码，输出一个包含运动信息的隐状态。
      将标准 LSTM 的矩阵乘法替换为卷积，保留 2D 空间结构。

ConvLSTM vs LSTM:
    LSTM:      x(t) 和 h(t-1) 拼接 → 全连接×4 → 4个门的标量向量
    ConvLSTM:  x[1×64×64] 和 h[32×64×64] 沿 channel 拼接[33×64×64]
               → 3×3 Conv → 4×32×64×64（每个空间位置有独立的门控）

Copy from: https://github.com/ndrplz/ConvLSTM_pytorch
本项目实际只用了 ConvLSTMCell（单层细胞），未使用多层 ConvLSTM。
"""

import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    """
    单个 ConvLSTM 细胞 —— 处理一个时间步的输入。

    内部状态:
        h (hidden state):  隐状态，作为"短期记忆"输出
        c (cell state):    细胞状态，作为"长期记忆"在时间步之间传递

    门控机制 (4个门，各 hidden_dim 通道):
        i (input gate):     输入门  —— 哪些新信息写入 c
        f (forget gate):    遗忘门  —— 哪些旧信息从 c 中丢弃
        o (output gate):    输出门  —— c 中的哪些内容暴露给 h
        g (candidate gate): 候选记忆 —— 新信息的具体内容
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        参数:
            input_dim:  输入张量的通道数（本项目=1，占用栅格图）
            hidden_dim: 隐状态的通道数（本项目=32，ConvLSTM输出通道）
            kernel_size: 卷积核大小，如 (3,3)
            bias:       卷积是否使用偏置项
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # same padding：输出空间尺寸 = 输入空间尺寸
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # ★ 关键卷积层：输入(1+32)=33通道 → 输出 4×32=128 通道
        # 4 个门 (i, f, o, g) 各占 hidden_dim=32 通道，一次卷积同时算出
        # 这比 4 次单独卷积高效
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,  # 33 = x的1 + h的32
            out_channels=4 * self.hidden_dim,              # 128 = 4个门 × 32通道
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        单个时间步的前向传播。

        参数:
            input_tensor: 当前帧输入，形状 (batch, input_dim=1, H, W)
            cur_state:   [h_cur, c_cur]，各形状 (batch, hidden_dim=32, H, W)

        返回:
            h_next: 更新后的隐状态 (batch, 32, H, W)
            c_next: 更新后的细胞状态 (batch, 32, H, W)
        """
        # 解包当前状态
        h_cur, c_cur = cur_state

        # 沿 channel 维度拼接输入和隐状态
        # input_tensor: (B, 1, H, W), h_cur: (B, 32, H, W)
        # → combined: (B, 33, H, W)
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 一次 3×3 卷积，输出 128 通道
        combined_conv = self.conv(combined)

        # 将 128 通道均分为 4 份，各 32 通道 → 4 个门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 门控激活：
        # i, f, o 用 sigmoid → 输出范围 (0, 1)，表示"开关"程度
        # g 用 tanh → 输出范围 (-1, 1)，表示候选记忆内容
        i = torch.sigmoid(cc_i)   # 输入门：1=写入，0=忽略
        f = torch.sigmoid(cc_f)   # 遗忘门：1=保留，0=丢弃
        o = torch.sigmoid(cc_o)   # 输出门：1=暴露，0=隐藏
        g = torch.tanh(cc_g)      # 候选记忆：新的信息内容（可正可负）

        # ★ 细胞状态更新：遗忘旧的 + 写入新的
        # f * c_cur：用遗忘门选择性丢弃旧记忆
        # i * g：用输入门选择性写入新候选信息
        c_next = f * c_cur + i * g

        # ★ 隐状态更新：用输出门过滤细胞状态
        # tanh(c_next)：将细胞状态压缩到 (-1, 1)
        # o * ...：输出门控制哪些内容暴露给下一时间步/下一层
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        初始化隐状态和细胞状态为零张量。

        参数:
            batch_size: batch 大小
            image_size: (H, W) 栅格地图尺寸，本项目为 (64, 64)

        返回:
            h0: 全零隐状态 (batch, 32, H, W)
            c0: 全零细胞状态 (batch, 32, H, W)
        """
        height, width = image_size
        # 初始化为全零——没有任何先验信息
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width,
                       device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width,
                       device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM —— 将多个 ConvLSTMCell 堆叠起来。

    注意：本项目实际上只用了 ConvLSTMCell，未使用此多层版本。
         这里保留仅为完整性。

    参数:
        input_dim:  输入通道数
        hidden_dim: 隐状态通道数（每层可不同）
        kernel_size: 卷积核大小
        num_layers: LSTM 层数
        batch_first: 是否 batch 在第0维
        bias:       是否使用偏置
        return_all_layers: 是否返回所有层的输出

    输入:
        5D Tensor: (B, T, C, H, W) 或 (T, B, C, H, W)
    输出:
        layer_output_list:  每层的输出序列列表
        last_state_list:    每层最后的状态 [(h, c), ...]
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # 扩展为多层：每层可以有不同的 hidden_dim 和 kernel_size
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 构建每一层：第i层输入通道 = (i==0 ? 原始输入 : 上一层隐状态通道)
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        多层 ConvLSTM 前向传播。

        参数:
            input_tensor: 5D Tensor (B, T, C, H, W) 或 (T, B, C, H, W)
            hidden_state: 可选的初始状态

        返回:
            layer_output_list: 每层所有时间步的输出
            last_state_list:   每层最后时间步的 (h, c)
        """
        # 统一为 batch_first = True 格式: (B, T, C, H, W)
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # 初始化或使用给定的隐状态
        if hidden_state is not None:
            raise NotImplementedError()  # 本项目未实现有状态模式
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 时间步数
        cur_layer_input = input_tensor   # 第一层的输入 = 原始输入

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # 该层的初始状态
            output_inner = []

            # 逐时间步处理
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],  # 第 t 帧
                    cur_state=[h, c]
                )
                output_inner.append(h)

            # 将该层所有时间步的输出堆叠起来: (B, T, C, H, W)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output  # 当前层输出 = 下一层输入

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # 默认只返回最后一层的输出
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """为所有层初始化全零隐状态。"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """检查 kernel_size 参数格式是否正确（必须是 tuple 或 tuple 的列表）。"""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and
                 all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """如果参数是单值，复制为 num_layers 长度的列表（每层相同参数）。"""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
