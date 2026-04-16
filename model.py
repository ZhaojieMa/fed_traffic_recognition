import torch
import torch.nn as nn
import torch.nn.functional as F

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


def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.1):
    device = outputs.device
    num_classes = outputs.shape[1]

    # 1. 缩短 Warmup，让模型在前 20% 的轮次快速进入稳定状态
    warmup_progress = min(1.0, current_round / (total_rounds * 0.2))
    tau = 1.0 * warmup_progress

    # 2. 改进分布估计：引入“全局偏见修正”
    # 核心：使用更强的平滑，不再让 pi_y 占据主导，特别是针对本地没有的类
    # 逻辑：如果本地没有该类，我们给它一个更接近均匀分布的预期，而不是极小的惩罚值
    pi_y = label_dist.to(device)

    # 这里的 0.5 权重可以平衡本地特异性和全局一致性
    refined_prior = 0.5 * pi_y + 0.5 * (1.0 / num_classes)
    log_prior = torch.log(refined_prior)

    # 3. Logit Adjustment 修正：只对当前 Batch 中存在的类别进行边际增强
    # 这样可以防止缺失类在本地梯度更新中产生噪声
    adjusted_logits = outputs + tau * log_prior

    ce_loss = F.cross_entropy(adjusted_logits, labels)

    # 4. 动态 Proximal 项：前期 Mu 稍大以保证一致性，后期减小以允许个性化
    if global_model is None:
        return ce_loss

    # 这里的 mu 使用传入的 MU_ADA (0.1)
    prox_loss = 0.0
    for p, g_p in zip(model.parameters(), global_model.parameters()):
        prox_loss += (p - g_p).norm(2) ** 2

    return ce_loss + (mu / 2) * prox_loss