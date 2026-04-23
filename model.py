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


def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.01):
    """
    本文设计的 FedLC-Ada 损失函数
    1. Logit Adjustment (LA): 解决 Non-IID 导致的类别频率偏移
    2. Proximal Term (FedProx): 限制本地模型对异构数据的过度拟合
    """
    device = outputs.device
    # 动态温度系数：随训练轮次增加逐步强化 Logit 补偿
    tau = current_round / total_rounds

    # pi_y: 本地类别的先验概率分布
    pi_y = torch.clamp(label_dist.to(device), min=1e-6)

    # $$ L_{comp} = f(x) + \tau \cdot \log(\pi_y) $$
    margin = tau * torch.log(pi_y)
    adjusted_outputs = outputs + margin

    ce_loss = F.cross_entropy(adjusted_outputs, labels)

    # Proximal Term
    prox_loss = 0
    if global_model is not None:
        for p, g_p in zip(model.parameters(), global_model.parameters()):
            prox_loss += (p - g_p).norm(2) ** 2

    return ce_loss + (mu / 2) * prox_loss


# 为了兼容性保留 FedProx 实现作为对照
def fedprox_loss(outputs, labels, model, global_model, mu=0.01):
    ce_loss = F.cross_entropy(outputs, labels)
    if global_model is None: return ce_loss
    prox_loss = sum((p - g_p).norm(2) ** 2 for p, g_p in zip(model.parameters(), global_model.parameters()))
    return ce_loss + (mu / 2) * prox_loss