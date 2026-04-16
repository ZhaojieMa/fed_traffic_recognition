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

    # 1. 动态温控 Tau (更激进的调整)
    progress = current_round / total_rounds
    tau = 1.0 if progress < 0.2 else (0.5 * (1.0 + math.cos(progress * math.pi)))

    # 2. 修正分布平滑：避免 log(0)，并引入全局先验补偿
    # label_dist 是本地频率。对于本地缺失类，赋予一个极小的 epsilon
    pi_y = label_dist.to(device)
    pi_y = torch.clamp(pi_y, min=1e-6)

    # 3. Logit Adjustment 核心公式修复
    # 逻辑：本地越少的类，惩罚越大，强制模型在全局聚合时更关注这些类的特征
    logit_adjustment = tau * torch.log(pi_y)
    adjusted_outputs = outputs + logit_adjustment

    ce_loss = F.cross_entropy(adjusted_outputs, labels)

    # 4. 配合 FedProx 的近端项防止本地模型跑飞
    prox_loss = 0
    if global_model is not None:
        for p, g_p in zip(model.parameters(), global_model.parameters()):
            prox_loss += (p - g_p).norm(2) ** 2

    return ce_loss + (mu / 2) * prox_loss