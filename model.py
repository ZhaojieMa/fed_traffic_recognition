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

def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.1):
    device = outputs.device
    num_classes = outputs.shape[1]

    # ================= 1. 余弦温启动策略 =================
    # 前 30% 的轮次平滑启动，解决原版前期表现垫底的问题
    progress = min(1.0, current_round / (total_rounds * 0.3))
    # 使用余弦函数实现 0 到 1 的丝滑过渡
    tau = 0.5 * (1.0 - math.cos(progress * math.pi))

    # ================= 2. 动态退火分布平滑 =================
    pi_y = label_dist.to(device)
    # smooth_factor 随训练进行从 0.6 线性衰减到 0.1
    # 前期强平滑抑制噪声，后期弱平滑信任真实的长尾分布
    smooth_factor = 0.5 * (1.0 - progress) + 0.1
    refined_prior = (1.0 - smooth_factor) * pi_y + smooth_factor * (1.0 / num_classes)
    log_prior = torch.log(refined_prior + 1e-8)

    # ================= 3. Logit 调整 =================
    adjusted_logits = outputs + tau * log_prior
    ce_loss = F.cross_entropy(adjusted_logits, labels)

    # ================= 4. 自适应 Proximal 约束 =================
    if global_model is None:
        return ce_loss

    # 核心创新：在 mu 的基础值上进行退火。
    # 前期约束强 (mu_t ≈ mu)，强制模型对齐全局；后期约束弱 (mu_t ≈ mu * 0.5)，允许模型个性化学习长尾特征
    dynamic_mu = mu * (1.0 - 0.5 * (current_round / total_rounds))

    prox_loss = 0.0
    for p, g_p in zip(model.parameters(), global_model.parameters()):
        prox_loss += (p - g_p).norm(2) ** 2

    return ce_loss + (dynamic_mu / 2) * prox_loss