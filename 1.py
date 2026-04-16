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
# import json
# import os
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# def extract_flow_features_from_json(json_path, label):
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         all_flows = []
#         for flow_id, flow_content in data.items():
#             flow_feats = flow_content.get('flow_features', {})
#             row = {}
#
#             for feat_name, directions in flow_feats.items():
#                 if 'biflow' in directions:
#                     for stat_name, value in directions['biflow'].items():
#                         if isinstance(value, (int, float)):
#                             row[f"{feat_name}_{stat_name}"] = value
#
#             if row:
#                 row['label'] = label
#                 all_flows.append(row)
#
#         df = pd.DataFrame(all_flows)
#         return df
#     except Exception as e:
#         print(f"处理 JSON 文件出错 {json_path}：{str(e)}")
#         return pd.DataFrame()
#
#
# if __name__ == "__main__":
#     # ====================== 只需改这里 ======================
#     folder_path = r"D:\MIRAGE-2019_traffic_dataset\Mi5_38_a4_ed_18_cc_bf"
#     # ========================================================
#
#     all_features = []
#     count = 0
#     # 自动遍历文件夹中所有 .json 文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json"):
#             if count == 100:
#                 break
#             count += 1
#             json_path = os.path.join(folder_path, filename)
#
#             # 自动用文件名作为标签（也可以自己映射）
#             label = filename  # 你也可以改成固定标签，如 "VPN"
#             print(f"正在处理：{filename}")
#
#             df = extract_flow_features_from_json(json_path, label)
#             if not df.empty:
#                 all_features.append(df)
#
#     if all_features:
#         final_df = pd.concat(all_features, ignore_index=True)
#         final_df = final_df.replace([float('inf'), float('-inf')], float('nan'))
#         final_df = final_df.fillna(0.0)
#
#         os.makedirs("./dataset", exist_ok=True)
#         final_df.to_csv("./dataset/traffic_features.csv", index=False, encoding='utf-8')
#         print(f"\n特征提取完成！")
#         print(f"总样本数：{len(final_df)}")
#         print(f"特征维度：{final_df.shape[1] - 1}")
#     else:
#         print("未提取到任何特征，请检查文件夹路径。")
