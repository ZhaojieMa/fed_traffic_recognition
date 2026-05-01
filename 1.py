# import os
# import random
# import json
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score, f1_score
# import flwr as fl
# from collections import OrderedDict
#
# # 确保已经按照之前的建议修改了 model.py
# from model import TrafficTransformer, fedprox_loss, fedlc_ada_loss
#
#
# # ==========================================
# # 1. 实验严谨性控制：全局随机种子固定
# # ==========================================
# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# seed_everything(42)
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ==========================================
# # 2. 全局超参数与元数据
# # ==========================================
# with open("./dataset/meta.json", "r") as f:
#     META = json.load(f)
# INPUT_DIM = META["input_dim"]
# NUM_CLASSES = META["num_classes"]
#
# NUM_CLIENTS = 10
# EPOCHS_PER_ROUND = 5
# TOTAL_ROUNDS = 30
# BATCH_SIZE = 32
# # Transformer 推荐更小的学习率，且加入 Weight Decay 防止 Non-IID 下过拟合
# LEARNING_RATE = 0.0005
# # 【学术严谨性】：MU设为0.5，显著抑制 Non-IID 带来的权重偏离
# MU = 0.5
#
#
# # ==========================================
# # 3. 数据加载逻辑
# # ==========================================
# def load_global_test():
#     """加载全局测试集，用于 Server 端评估"""
#     df = pd.read_csv("./dataset/global_test.csv")
#     X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32).to(DEVICE)
#     y = torch.tensor(df['label'].values, dtype=torch.long).to(DEVICE)
#     return X, y
#
#
# GLOBAL_X_TEST, GLOBAL_Y_TEST = load_global_test()
#
#
# def load_client_data(client_id, alpha):
#     """加载本地客户端数据及其类别分布"""
#     data_path = f"./dataset/non_iid_alpha_{alpha}/client_{client_id}.csv"
#     dist_path = f"./dataset/non_iid_alpha_{alpha}/client_{client_id}_dist.npy"
#
#     df = pd.read_csv(data_path)
#     X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32).to(DEVICE)
#     y = torch.tensor(df['label'].values, dtype=torch.long).to(DEVICE)
#
#     # 类别分布已在 non_iid_split.py 中通过 Laplace 平滑处理
#     label_dist = torch.tensor(np.load(dist_path), dtype=torch.float32).to(DEVICE)
#     return X, y, label_dist
#
#
# # ==========================================
# # 4. 联邦学习客户端定义
# # ==========================================
# class TrafficClient(fl.client.NumPyClient):
#     def __init__(self, client_id, alpha, method):
#         self.method = method
#         self.X_train, self.y_train, self.label_dist = load_client_data(client_id, alpha)
#         self.model = TrafficTransformer(INPUT_DIM, NUM_CLASSES).to(DEVICE)
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
#
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]
#
#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)
#
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         current_round = config.get("server_round", 1)
#
#         # 为 FedProx 和本文方法准备全局模型副本
#         global_model = None
#         if self.method in ["FedProx", "Proposed"]:
#             global_model = TrafficTransformer(INPUT_DIM, NUM_CLASSES).to(DEVICE)
#             global_model.load_state_dict(self.model.state_dict())
#             global_model.eval()
#
#         self.model.train()
#         loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=BATCH_SIZE, shuffle=True)
#
#         for _ in range(EPOCHS_PER_ROUND):
#             for bx, by in loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(bx)
#
#                 if self.method == "Proposed":
#                     # 本文方法：基于 FedProx 叠加 Logit Calibration 与自适应 Focal 权重
#                     loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist, current_round,
#                                           TOTAL_ROUNDS, mu=MU)
#                 elif self.method == "FedProx":
#                     loss = fedprox_loss(outputs, by, self.model, global_model, mu=MU)
#                 else:
#                     loss = F.cross_entropy(outputs, by)
#
#                 loss.backward()
#                 self.optimizer.step()
#
#         return self.get_parameters(config), len(self.X_train), {}
#
#
# # ==========================================
# # 5. 评价函数与启动逻辑
# # ==========================================
# def get_evaluate_fn():
#     def evaluate(server_round, parameters, config):
#         model = TrafficTransformer(INPUT_DIM, NUM_CLASSES).to(DEVICE)
#         params_dict = zip(model.state_dict().keys(), parameters)
#         model.load_state_dict(OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict}))
#         model.eval()
#         with torch.no_grad():
#             outputs = model(GLOBAL_X_TEST)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             true_y = GLOBAL_Y_TEST.cpu().numpy()
#             acc = float(accuracy_score(true_y, preds))
#             f1 = float(f1_score(true_y, preds, average='macro'))
#         return 0.0, {"accuracy": acc, "f1": f1}
#
#     return evaluate
#
#
# def run_experiment(method, alpha):
#     print(f"\n[实验启动] 方法: {method} | 数据异构度 α: {alpha}")
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         evaluate_fn=get_evaluate_fn(),
#         on_fit_config_fn=lambda r: {"server_round": r}
#     )
#
#     # 使用 Flower 仿真框架
#     history = fl.simulation.start_simulation(
#         client_fn=lambda cid: TrafficClient(int(cid), alpha, method).to_client(),
#         num_clients=NUM_CLIENTS,
#         config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
#         strategy=strategy,
#         client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0}
#     )
#
#     final_acc = history.metrics_centralized["accuracy"][-1][1]
#     final_f1 = history.metrics_centralized["f1"][-1][1]
#     return final_acc, final_f1
#
#
# # ==========================================
# # 6. 集中式训练 (对比上限)
# # ==========================================
# def centralized_baseline(alpha):
#     print(f"\n[基准] 运行集中式训练 (α={alpha})")
#     all_x, all_y = [], []
#     for i in range(NUM_CLIENTS):
#         x, y, _ = load_client_data(i, alpha)
#         all_x.append(x);
#         all_y.append(y)
#
#     loader = DataLoader(TensorDataset(torch.cat(all_x), torch.cat(all_y)), batch_size=BATCH_SIZE, shuffle=True)
#     model = TrafficTransformer(INPUT_DIM, NUM_CLASSES).to(DEVICE)
#     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#
#     model.train()
#     for _ in range(50):  # 集中式训练收敛快，50轮足够
#         for bx, by in loader:
#             optimizer.zero_grad()
#             loss = F.cross_entropy(model(bx), by)
#             loss.backward()
#             optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         preds = torch.argmax(model(GLOBAL_X_TEST), dim=1).cpu().numpy()
#         acc = accuracy_score(GLOBAL_Y_TEST.cpu().numpy(), preds)
#         f1 = f1_score(GLOBAL_Y_TEST.cpu().numpy(), preds, average='macro')
#     return float(acc), float(f1)
#
#
# # ==========================================
# # 新增：本地独立训练逻辑
# # ==========================================
# def local_only_training(alpha):
#     """模拟没有任何联邦通信的情况，客户端仅在本地数据训练，并在全局测试集评估"""
#     print(f"\n[基准] 运行本地独立训练 (α={alpha})")
#     all_client_accs = []
#     all_client_f1s = []
#
#     for i in range(NUM_CLIENTS):
#         X, y, _ = load_client_data(i, alpha)
#         # 保持模型结构一致
#         model = TrafficTransformer(INPUT_DIM, NUM_CLASSES).to(DEVICE)
#         optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
#
#         # 训练总时长与联邦对齐: TOTAL_ROUNDS * EPOCHS_PER_ROUND
#         loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
#         model.train()
#         for _ in range(TOTAL_ROUNDS * EPOCHS_PER_ROUND):
#             for bx, by in loader:
#                 optimizer.zero_grad()
#                 F.cross_entropy(model(bx), by).backward()
#                 optimizer.step()
#
#         model.eval()
#         with torch.no_grad():
#             preds = torch.argmax(model(GLOBAL_X_TEST), dim=1).cpu().numpy()
#             true_y = GLOBAL_Y_TEST.cpu().numpy()
#             all_client_accs.append(accuracy_score(true_y, preds))
#             all_client_f1s.append(f1_score(true_y, preds, average='macro'))
#
#     # 返回所有客户端表现的平均值作为该 α 下的本地训练基准
#     return float(np.mean(all_client_accs)), float(np.mean(all_client_f1s))
#
#
# # ==========================================
# # 7. 主程序入口
# # ==========================================
# if __name__ == "__main__":
#     os.makedirs("./results", exist_ok=True)
#     # alpha = 0.1 表示强 Non-IID，最能体现算法优越性
#     alphas = [0.1, 0.5]
#     summary = {}
#
#     for alpha in alphas:
#         print(f"\n{'#' * 30}\n正在测试异构度 α = {alpha}\n{'#' * 30}")
#
#         # 1. 本地独立训练 (新加)
#         l_acc, l_f1 = local_only_training(alpha)
#
#         # 2. 集中式上限
#         c_acc, c_f1 = centralized_baseline(alpha)
#
#         # 3. FedAvg
#         a_acc, a_f1 = run_experiment("FedAvg", alpha)
#
#         # 4. FedProx
#         p_acc, p_f1 = run_experiment("FedProx", alpha)
#
#         # 5. 本文 Proposed (FedLC-Ada)
#         o_acc, o_f1 = run_experiment("Proposed", alpha)
#
#         # 组织数据，确保与绘图脚本顺序匹配
#         summary[str(alpha)] = {
#             "methods": ["本地独立", "FedAvg", "FedProx", "本文方法(FedLC-Ada)", "集中式(上限)"],
#             "accuracies": [l_acc, a_acc, p_acc, o_acc, c_acc],
#             "f1_scores": [l_f1, a_f1, p_f1, o_f1, c_f1]
#         }
#
#     with open("./results/metrics.json", "w", encoding='utf-8') as f:
#         json.dump(summary, f, indent=4, ensure_ascii=False)
#
#     print("\n[完成] 所有对比实验数据（含本地训练）已存至 ./results/metrics.json，请运行 analysis.py 绘图。")

# import pandas as pd
# import numpy as np
# import os
# import json
# import warnings
# from sklearn.preprocessing import StandardScaler
#
# warnings.filterwarnings('ignore')
#
# # 精确对齐 Darknet.CSV 的特征
# TARGET_FEATURES = [
#     'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
#     'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
#     'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
#     'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
#     'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
#     'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
#     'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
#     'Fwd PSH Flags', 'Fwd Header Length', 'Bwd Header Length',
#     'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
#     'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
#     'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
#     'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
# ]
#
# import pandas as pd
# import numpy as np
# import os
# import json
# from sklearn.preprocessing import StandardScaler
#
# # 根据 CIC-Darknet2020 论文标准，需要剔除的非特征列（防止过拟合/特征泄露）
# DROP_COLUMNS = [
#     'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
#     'Destination Port', 'Timestamp', 'Label'  # 这里的 Label 是大类，我们通常预测 Label.1
# ]
#
#
# def preprocess_darknet_csv(file_path):
#     print(f"正在深度预处理文件: {file_path} ...")
#     df = pd.read_csv(file_path, low_memory=False)
#
#     # 1. 动态检测标签列并清洗
#     # CIC-Darknet2020 的 Label.1 对应具体的流量类型 (P2P, Audio-Streaming, etc.)
#     label_col = 'Label.1' if 'Label.1' in df.columns else df.columns[-1]
#     df[label_col] = df[label_col].astype(str).str.strip().str.lower()
#
#     # 【新增】剔除无意义或导致泄露的列
#     cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns and c != label_col]
#     df = df.drop(columns=cols_to_drop)
#
#     # 2. 自动特征筛选（剔除非数值列和标签列）
#     # 相比硬编码 TARGET_FEATURES，这样更具鲁棒性
#     features = df.drop(columns=[label_col]).copy()
#     labels = df[label_col].copy()
#
#     # 3. 数据清洗：强制数值化并处理 Inf/NaN
#     for col in features.columns:
#         if features[col].dtype == 'object':
#             features[col] = pd.to_numeric(features[col], errors='coerce')
#
#     # 替换 Inf 为 NaN 并用 0 填充
#     features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
#
#     # 【新增】剔除全 0 或常数列（无区分度特征）
#     features = features.loc[:, (features != features.iloc[0]).any()]
#
#     # 4. 【核心优化】构造比例特征 (增加区分度)
#     # 注意：确保这些基础列在清洗后依然存在
#     fwd_pkt_col = 'Total Fwd Packets' if 'Total Fwd Packets' in features.columns else 'Total Fwd Packet'
#     bwd_pkt_col = 'Total Bwd packets'
#     fwd_len_col = 'Total Length of Fwd Packets' if 'Total Length of Fwd Packets' in features.columns else 'Total Length of Fwd Packet'
#     bwd_len_col = 'Total Length of Bwd Packets' if 'Total Length of Bwd Packets' in features.columns else 'Total Length of Bwd Packet'
#
#     if fwd_pkt_col in features.columns and bwd_pkt_col in features.columns:
#         features['fwd_bwd_ratio'] = features[fwd_pkt_col] / (features[bwd_pkt_col] + 1)
#         features['avg_byte_per_pkt'] = (features[fwd_len_col] + features[bwd_len_col]) / \
#                                        (features[fwd_pkt_col] + features[bwd_pkt_col] + 1)
#
#     # 5. 【核心优化】Log1p + StandardScaler 组合拳
#     # 先做 Log 变换平滑流量数据的长尾分布（流量数据通常服从幂律分布）
#     features = np.log1p(features.clip(lower=0))
#
#     # 再做标准化
#     scaler = StandardScaler()
#     scaled_values = scaler.fit_transform(features)
#     features_df = pd.DataFrame(scaled_values, columns=features.columns)
#
#     print(f"预处理完成。原始特征数: {df.shape[1]}, 最终保留特征数: {features_df.shape[1]}")
#     return features_df, labels, list(features_df.columns)
#
#
# if __name__ == "__main__":
#     # 建议使用绝对路径或确保文件存在
#     csv_path = "D:/Darknet.CSV"
#
#     if os.path.exists(csv_path):
#         features_df, raw_labels, final_feature_list = preprocess_darknet_csv(csv_path)
#
#         # 标签转 ID（确保类别一致性）
#         unique_labels = sorted(raw_labels.unique())
#         label_to_id = {label: i for i, label in enumerate(unique_labels)}
#
#         # 使用 loc 确保索引对齐
#         features_df['label'] = raw_labels.map(label_to_id).reset_index(drop=True)
#
#         # 【新增】处理重复行（防止数据泄露）
#         initial_count = len(features_df)
#         features_df = features_df.drop_duplicates().reset_index(drop=True)
#         print(f"剔除重复流: {initial_count} -> {len(features_df)}")
#
#         # 保存
#         os.makedirs("./dataset", exist_ok=True)
#         features_df.to_csv("./dataset/traffic_features.csv", index=False)
#
#         with open("./dataset/meta.json", "w") as f:
#             json.dump({
#                 "input_dim": len(final_feature_list),
#                 "num_classes": len(label_to_id),
#                 "class_map": label_to_id
#             }, f, indent=4)
#
#         print(f"数据集已保存，维度: {len(final_feature_list)}, 类别数: {len(label_to_id)}")
#     else:
#         print(f"错误: 找不到 {csv_path}")


# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channels, max(1, channels // reduction), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(1, channels // reduction), channels, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         w = self.fc(x)
#         return x * w
#
#
# class TrafficResNet(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(TrafficResNet, self).__init__()
#
#         # 增加宽度到 512 解决瓶颈
#         self.hidden_dim = 512
#
#         self.input_layer = nn.Sequential(
#             nn.Linear(input_dim, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim),
#             nn.ReLU()
#         )
#
#         # 残差块 1 (512 -> 512)
#         self.res1 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.se1 = SEBlock(self.hidden_dim)
#
#         # 残差块 2 (512 -> 512)
#         # 注意：这里必须也是 512，否则会发生你遇到的 mat1/mat2 报错
#         self.res2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.se2 = SEBlock(self.hidden_dim)
#
#         # 如果 res2 的输出维度和输入维度一致，就不需要 proj2
#         # 但为了代码健壮性，我们可以保留一个恒等映射或线性层
#         self.proj2 = nn.Identity()
#
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(self.hidden_dim, num_classes)
#
#     def forward(self, x):
#         x = self.input_layer(x)
#
#         # Residual 1
#         identity = x
#         x = F.relu(self.res1(x))
#         x = self.se1(x) + identity
#
#         # Residual 2
#         identity = self.proj2(x)
#         x = F.relu(self.res2(x))
#         x = self.se2(x) + identity
#
#         return self.classifier(self.dropout(x))
# def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.01):
#     device = outputs.device
#
#     # 1. 原有的 Logit Adjustment (保持不变)
#     tau = current_round / total_rounds
#     pi_y = torch.clamp(label_dist.to(device), min=1e-6)
#     margin = tau * torch.log(pi_y)
#     adjusted_outputs = outputs + margin
#     ce_loss = F.cross_entropy(adjusted_outputs, labels)
#
#     # --- 创新点：动态梯度重加权 (DGR) ---
#     # 计算当前 Batch 的“稀缺性因子” (Scarcity Factor)
#     # 统计当前 Batch 中每个类别的出现频率
#     batch_size = labels.size(0)
#     unique_labels, counts = labels.unique(return_counts=True)
#     batch_dist = torch.zeros_like(pi_y)
#     batch_dist[unique_labels] = counts.float() / batch_size
#
#     # 稀缺性因子 S：本地 Batch 中“长尾类别”的占比
#     # 这里我们假设频率低于平均值的类别是长尾类别
#     avg_freq = pi_y.mean()
#     # S 越大，说明当前 Batch 包含越多全局稀缺的长尾样本
#     S = (batch_dist * (pi_y < avg_freq).float()).sum().item()
#
#     # --- 动态调整 Proximal 的权重 ---
#     # 基础 Proximal Loss
#     prox_loss = 0
#     if global_model is not None:
#         for p, g_p in zip(model.parameters(), global_model.parameters()):
#             prox_loss += (p - g_p).norm(2) ** 2
#
#     # 动态系数 Lambda: 当 S (稀缺样本占比) 高时，降低 Proximal 的影响
#     # 这样模型在遇到难得一见的长尾样本时，可以更自由地更新参数，而不被全局模型锁死
#     # alpha_dgr 是一个超参数，控制动态范围
#     alpha_dgr = 0.5
#     lambda_prox = 1.0 - alpha_dgr * S  # S 越大，lambda 越小
#
#     # --- 最终 Loss 组合 ---
#     # 注意：这里不再是简单的加法，而是受 S 调制的加法
#     total_loss = ce_loss + (mu / 2) * lambda_prox * prox_loss
#
#     return total_loss


# def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds,
#                    mu=0.01, gamma=2.0, use_la=True, use_focal=True, use_decoupled_prox=True):
#     device = outputs.device
#
#     # ---------------------------------------------------------
#     # 1. Logit Adjustment (LA)
#     # ---------------------------------------------------------
#     if use_la:
#         # 优化：引入退火系数，随轮数增加逐渐增强调整强度
#         tau = (current_round / total_rounds) ** 2
#         pi_y = torch.clamp(label_dist.to(device), min=1e-4)
#         margin = tau * torch.log(pi_y)
#         adjusted_outputs = outputs + margin
#     else:
#         adjusted_outputs = outputs
#
#     # ---------------------------------------------------------
#     # 2. Focal Loss
#     # ---------------------------------------------------------
#     if use_focal:
#         probs = F.softmax(adjusted_outputs, dim=1)
#         target_probs = probs[range(labels.size(0)), labels]
#         focal_weight = (1 - target_probs) ** gamma
#         ce_loss = F.cross_entropy(adjusted_outputs, labels, reduction='none')
#         task_loss = (focal_weight * ce_loss).mean()
#     else:
#         task_loss = F.cross_entropy(adjusted_outputs, labels)
#
#     # ---------------------------------------------------------
#     # 3. 改进的 Proximal Term (核心改进点)
#     # ---------------------------------------------------------
#     prox_loss = 0.0
#     if global_model is not None and mu > 0:
#         # 计算本地类别的相对频率分布，归一化到 [0, 1]
#         # 加 1e-8 防止除以 0
#         local_freq = label_dist.to(device)
#         norm_freq = local_freq / (local_freq.max() + 1e-8)
#
#         for (name, p), (_, g_p) in zip(model.named_parameters(), global_model.named_parameters()):
#             if use_decoupled_prox:
#                 # ==========================================
#                 # 分类头：进行【类别维度】的细粒度解耦
#                 # ==========================================
#                 if 'classifier.weight' in name and p.shape[0] == len(label_dist):
#                     # p 的形状为 [num_classes, in_features]
#                     # 权重惩罚系数：频率越高，约束越小；频率越低（甚至为0），约束越大(接近 mu * 2.0)
#                     # 形状调整为 [num_classes, 1] 以便利用广播机制
#                     class_penalty = mu * (1.0 + (1.0 - norm_freq)).view(-1, 1)
#
#                     # 逐元素计算 L2 并应用类别感知的 penalty
#                     layer_loss = (class_penalty * (p - g_p) ** 2).sum()
#                     prox_loss += layer_loss
#
#                 elif 'classifier.bias' in name and p.shape[0] == len(label_dist):
#                     class_penalty = mu * (1.0 + (1.0 - norm_freq))
#                     layer_loss = (class_penalty * (p - g_p) ** 2).sum()
#                     prox_loss += layer_loss
#
#                 # ==========================================
#                 # 特征提取层：保持统一的较强约束，维持全局特征对齐
#                 # ==========================================
#                 else:
#                     prox_loss += (mu * 1.0) * (p - g_p).norm(2) ** 2
#             else:
#                 # 标准 FedProx: 无差别全量约束
#                 prox_loss += mu * (p - g_p).norm(2) ** 2
#
#     return task_loss + 0.5 * prox_loss
#
#     def fedlc_ada_loss(outputs, labels, model, global_model, label_dist, current_round, total_rounds, mu=0.01,
#                        gamma=2.0,
#                        prox_mode='decoupled'):
#         """
#         带消融实验开关的 FedLC-Ada 损失函数
#         prox_mode 可选: 'none', 'standard', 'decoupled'
#         """
#         device = outputs.device
#         tau = (current_round / total_rounds) ** 2
#         pi_y = torch.clamp(label_dist.to(device), min=1e-4)
#
#         margin = tau * torch.log(pi_y)
#         adjusted_outputs = outputs + margin
#
#         probs = F.softmax(adjusted_outputs, dim=1)
#         target_probs = probs[range(labels.size(0)), labels]
#         focal_weight = (1 - target_probs) ** gamma
#         ce_loss = F.cross_entropy(adjusted_outputs, labels, reduction='none')
#         task_loss = (focal_weight * ce_loss).mean()
#
#         # 近端项消融逻辑
#         prox_loss = 0.0
#         if global_model is not None and prox_mode != 'none':
#             for (name, p), (_, g_p) in zip(model.named_parameters(), global_model.named_parameters()):
#                 if prox_mode == 'decoupled':
#                     # 本文提出的解耦模式：释放分类头
#                     layer_mu = mu * 0.1 if 'classifier' in name else mu
#                 else:
#                     # 标准 FedProx 模式：全局死板约束
#                     layer_mu = mu
#
#                 prox_loss += layer_mu * (p - g_p).norm(2) ** 2
#
#         return task_loss + 0.5 * prox_loss
def fedlc_ada_loss(outputs, labels, model, global_model, label_dist,
                   current_round, total_rounds,
                   mu=0.001, gamma=2.0,
                   use_la=True, use_focal=True, use_decoupled_prox=True):

    device = outputs.device
    tau = current_round / total_rounds

    # =========================
    # ✅ 1. 正确的 LA（用 batch 分布，不是 client 分布）
    # =========================
    if use_la:
        num_classes = outputs.shape[1]

        # 计算 batch 内真实分布（关键修复）
        batch_hist = torch.bincount(labels, minlength=num_classes).float().to(device)
        batch_dist = batch_hist / (batch_hist.sum() + 1e-8)

        margin = tau * torch.log(batch_dist + 1e-5)
        adjusted_outputs = outputs + margin
    else:
        adjusted_outputs = outputs

    # =========================
    # ✅ 2. Focal Loss（延迟启动，避免前期冻结）
    # =========================
    if use_focal:
        if current_round < total_rounds * 0.3:
            # 前30%轮：完全关闭 focal
            task_loss = F.cross_entropy(adjusted_outputs, labels)
        else:
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                pt = probs[range(labels.size(0)), labels]

            dynamic_gamma = gamma * (tau - 0.3) / 0.7  # 从0慢慢升到gamma
            focal_weight = (1.0 - pt) ** dynamic_gamma

            ce = F.cross_entropy(adjusted_outputs, labels, reduction='none')
            task_loss = (focal_weight * ce).mean()
    else:
        task_loss = F.cross_entropy(adjusted_outputs, labels)

    # =========================
    # ✅ 3. Decoupled Prox（强度大幅降低 + 延迟）
    # =========================
    prox_loss = 0.0

    if global_model is not None and mu > 0:

        # 🔥 关键：前40轮不加prox（否则直接压死）
        if current_round < total_rounds * 0.4:
            return task_loss

        for (name, p), (_, g_p) in zip(model.named_parameters(), global_model.named_parameters()):

            if use_decoupled_prox and 'classifier' in name:

                # 🔥 关键：弱化类别惩罚（否则直接崩）
                if 'weight' in name:
                    prox_loss += (p - g_p).pow(2).sum() * mu * 0.5
                else:
                    prox_loss += (p - g_p).pow(2).sum() * mu * 0.5

            else:
                prox_loss += mu * (p - g_p).pow(2).sum()

    return task_loss + 0.5 * prox_loss