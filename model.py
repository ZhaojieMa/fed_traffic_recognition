import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """特征注意力机制：自动增强关键流特征权重"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class TrafficResNet(nn.Module):
    """增强型残差全连接网络，防止深层网络欠拟合"""

    def __init__(self, input_dim, num_classes):
        super(TrafficResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 残差块 1
        self.res1 = nn.Linear(256, 256)
        self.se1 = SEBlock(256)

        # 残差块 2
        self.res2 = nn.Linear(256, 128)
        self.proj2 = nn.Linear(256, 128)  # 维度匹配
        self.se2 = SEBlock(128)

        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.input_layer(x)

        # Residual 1
        identity = x
        x = F.relu(self.res1(x))
        x = self.se1(x) + identity

        # Residual 2
        identity = self.proj2(x)
        x = F.relu(self.res2(x))
        x = self.se2(x) + identity

        return self.classifier(self.dropout(x))


def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds,
                   mu=0.01, gamma=2.0, use_la=True, use_focal=True, use_decoupled_prox=True):
    device = outputs.device

    # ---------------------------------------------------------
    # 1. Logit Adjustment (LA) - 线性平滑退火
    # ---------------------------------------------------------
    tau = current_round / total_rounds

    if use_la:
        pi_y = torch.clamp(label_dist.to(device), min=1e-5)
        margin = tau * torch.log(pi_y)
        adjusted_outputs = outputs + margin
    else:
        adjusted_outputs = outputs

    # ---------------------------------------------------------
    # 2. Focal Loss
    # ---------------------------------------------------------
    if use_focal:
        with torch.no_grad():
            clean_probs = F.softmax(outputs, dim=1)
            target_probs = clean_probs[range(labels.size(0)), labels]

        focal_weight = (1.0 - target_probs) ** gamma

        ce_loss = F.cross_entropy(adjusted_outputs, labels, reduction='none')
        task_loss = (focal_weight * ce_loss).mean()
    else:
        task_loss = F.cross_entropy(adjusted_outputs, labels)

    # ---------------------------------------------------------
    # 3. 解耦 Proximal Term
    # ---------------------------------------------------------
    prox_loss = 0.0
    if global_model is not None and mu > 0:
        device = outputs.device
        local_freq = label_dist.to(device)
        norm_freq = local_freq / (local_freq.max() + 1e-8)

        # 动态退火系数：随训练轮数增加，整体 Prox 约束从 1.0 降到 0.5
        # 目的：训练后期允许模型进行微调，解决 metrics_8.json 中后期的震荡和停滞
        tau_t = 1.0 - 0.5 * (current_round / total_rounds)

        for (name, p), (_, g_p) in zip(model.named_parameters(), global_model.named_parameters()):
            # 1. 针对分类头 (classifier.weight/bias)
            if use_decoupled_prox and 'classifier' in name:
                # 核心：逆频率保护（防止少数类被擦除）
                # 惩罚系数范围：少数类 ≈ 3*mu, 多数类 ≈ 1*mu
                class_penalty = mu * (1.0 + 2.0 * (1.0 - norm_freq)) * tau_t

                if 'weight' in name:
                    # 针对 Linear 层的 Weight 形状处理
                    prox_loss += (class_penalty.view(-1, 1) * (p - g_p) ** 2).sum()
                else:
                    prox_loss += (class_penalty * (p - g_p) ** 2).sum()

            # 2. 针对 Backbone (日志中的 input_layer, res, se, proj 等)
            else:
                # 改进点：对特征层使用更轻的约束 (0.1 * mu)
                # 理由：特征提取层需要灵活性来适应 Non-IID 数据
                prox_loss += (mu * 0.1 * tau_t) * (p - g_p).pow(2).sum()

    return task_loss + 0.5 * prox_loss


# 为了兼容性保留 FedProx 实现作为对照
def fedprox_loss(outputs, labels, model, global_model, mu=0.01):
    ce_loss = F.cross_entropy(outputs, labels)
    if global_model is None: return ce_loss
    prox_loss = sum((p - g_p).norm(2) ** 2 for p, g_p in zip(model.parameters(), global_model.parameters()))
    return ce_loss + (mu / 2) * prox_loss