import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrafficMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TrafficMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def fedprox_loss(outputs, labels, model, global_model, mu=0.1):
    ce_loss = F.cross_entropy(outputs, labels)
    if global_model is None: return ce_loss
    prox_loss = sum((p - g_p).norm(2) ** 2 for p, g_p in zip(model.parameters(), global_model.parameters()))
    return ce_loss + (mu / 2) * prox_loss


# model.py 修复部分
def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.1):
    device = outputs.device

    # ================= 核心修复 2：废除归零退火 =================
    # 在非独立同分布的长尾联邦学习中，一旦丢弃惩罚项，模型会立刻被本地大类重新带偏
    # 改为全程使用恒定的 tau=1.0，给少见类提供最强力的保护伞！
    tau = 1.0

    # 避免 log(0) 崩溃，1e-5 截断能提供约 -11.5 的强力惩罚对数值
    pi_y = label_dist.to(device)
    pi_y = torch.clamp(pi_y, min=1e-5)

    # Logit Adjustment 逻辑：本地越少的类，惩罚对数越小（绝对值越大），
    # 强迫模型在损失反馈时付出更大的梯度代价去拟合稀有类。
    logit_adjustment = tau * torch.log(pi_y)
    adjusted_outputs = outputs + logit_adjustment

    ce_loss = F.cross_entropy(adjusted_outputs, labels)

    prox_loss = 0
    if global_model is not None:
        for p, g_p in zip(model.parameters(), global_model.parameters()):
            prox_loss += (p - g_p).norm(2) ** 2

    return ce_loss + (mu / 2) * prox_loss