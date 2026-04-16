import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def simple_dirichlet_split(y, num_clients, alpha=0.5):
    """标准的 Dirichlet 划分（不加任何平滑和噪声）"""
    num_classes = len(np.unique(y))
    client_data_idx = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        if len(idx_k) == 0: continue
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(idx_k)).astype(int)

        diff = len(idx_k) - counts.sum()
        for i in range(diff): counts[i % num_clients] += 1

        current_pos = 0
        for i in range(num_clients):
            size = counts[i]
            client_data_idx[i].extend(idx_k[current_pos: current_pos + size].tolist())
            current_pos += size
    return client_data_idx


def make_global_long_tail(df, total_samples=10000, zipf_alpha=1.5):
    """
    使用 Zipf 定律构造更贴近真实互联网流量的长尾分布
    zipf_alpha: 越大幅度越极端，建议 1.2 - 1.5 之间
    """
    class_counts = df['label'].value_counts().sort_values(ascending=False)
    classes = class_counts.index.tolist()
    num_classes = len(classes)

    # 生成 Zipf 权重
    ranks = np.arange(1, num_classes + 1)
    weights = 1.0 / (ranks ** zipf_alpha)
    probabilities = weights / weights.sum()

    sampled_dfs = []
    for i, cls in enumerate(classes):
        # 计算该类理论应分得的样本数
        expected_n = int(total_samples * probabilities[i])

        # 获取该类实际拥有的样本数
        cls_df = df[df['label'] == cls]
        cls_available = len(cls_df)

        # 确定最终采样数：不能超过拥有数，且尾部类别至少保留3-5个样本以防消失
        n = max(min(expected_n, cls_available), 3)

        sampled_dfs.append(cls_df.sample(n=n, random_state=42))

    return pd.concat(sampled_dfs).sample(frac=1.0, random_state=42)


if __name__ == "__main__":
    csv_path = "./dataset/traffic_features.csv"
    if not os.path.exists(csv_path): exit("找不到数据")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != 'label']
    for col in feature_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'].astype(str))

    # 1. 划分基础训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # 归一化处理
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    test_df.to_csv("./dataset/global_test.csv", index=False)

    # ================= 动态样本量设定逻辑 =================
    # 建议论文使用量级：10000 - 20000 条。
    # 这里设定目标为 12000，或者原始训练集的 80%（防止数据集本身很小的情况）
    TARGET_TOTAL = 12000
    TOTAL_SAMPLES_TO_USE = min(TARGET_TOTAL, int(len(train_df) * 0.8))

    # 设定长尾系数：1.2 相比 1.5 更温和，样本量大时能让尾部类学到更多特征
    ZIPF_ALPHA = 1.5
    # ====================================================

    num_clients = 10
    alphas = [0.1]
    num_classes = len(le.classes_)

    # 1. 生成 RWTH (实验组：全局长尾异构数据池)
    train_df_rwth = make_global_long_tail(train_df, total_samples=TOTAL_SAMPLES_TO_USE, zipf_alpha=ZIPF_ALPHA)

    # 2. 生成 Simple (对照组：随机抽样同样的数量，保证公平对比)
    train_df_simple = train_df.sample(n=len(train_df_rwth), random_state=42)

    for alpha in alphas:
        # 分别在两个池子上执行 Dirichlet 划分
        splits = {
            "simple": (train_df_simple, simple_dirichlet_split(train_df_simple['label'].values, num_clients, alpha)),
            "rwth": (train_df_rwth, simple_dirichlet_split(train_df_rwth['label'].values, num_clients, alpha))
        }

        for split_type, (base_df, client_indices) in splits.items():
            save_dir = f"./dataset/{split_type}_alpha_{alpha}"
            os.makedirs(save_dir, exist_ok=True)

            for c in range(num_clients):
                client_df = base_df.iloc[client_indices[c]]
                client_df.to_csv(f"{save_dir}/client_{c}.csv", index=False)

                # 记录真实的本地分布：不加任何平滑！让0保持为0！
                counts = client_df['label'].value_counts()
                dist = np.zeros(num_classes)
                total_samples_client = len(client_df)

                for i in range(num_classes):
                    cnt = counts.get(i, 0)
                    dist[i] = cnt / (total_samples_client + 1e-9)

                dist = dist / dist.sum()
                np.save(f"{save_dir}/client_{c}_dist.npy", dist)

    with open("./dataset/meta.json", "w") as f:
        json.dump({
            "input_dim": len(feature_cols),
            "num_classes": num_classes,
            "classes": le.classes_.tolist(),
            "total_training_samples": len(train_df_rwth)
        }, f)

    print(f"数据划分成功！")
    print(f"原始数据总量: {len(df)}")
    print(f"最终实验选用的总样本量: {len(train_df_rwth)}")
    print(f"划分模式: Simple组 vs RWTH组(Zipf Alpha={ZIPF_ALPHA})")