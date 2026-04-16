import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def simple_dirichlet_split(y, num_clients, alpha=0.5):
    """标准的 Dirichlet 划分（对照组使用）"""
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


def realistic_traffic_split(y, num_clients, alpha=0.1, noise_ratio=0.03):
    """
    【本文核心构造】真实流量混合划分：高度个性化主应用 + 全局微量背景底噪
    noise_ratio: 提取 3% 作为全局共享背景流量
    """
    num_classes = len(np.unique(y))
    client_data_idx = [[] for _ in range(num_clients)]

    # 区分主体流量和背景噪声
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    noise_size = int(len(y) * noise_ratio)
    noise_idx = indices[:noise_size]
    main_idx = indices[noise_size:]

    # 1. 主体流量：使用极端的 Dirichlet (模拟用户的极端应用偏好)
    y_main = y[main_idx]
    for k in range(num_classes):
        idx_k = np.where(y_main == k)[0]
        real_idx_k = main_idx[idx_k]
        np.random.shuffle(real_idx_k)
        if len(real_idx_k) == 0: continue

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(real_idx_k)).astype(int)
        diff = len(real_idx_k) - counts.sum()
        for i in range(diff): counts[i % num_clients] += 1

        pos = 0
        for i in range(num_clients):
            client_data_idx[i].extend(real_idx_k[pos: pos + counts[i]].tolist())
            pos += counts[i]

    # 2. 背景噪声：均匀分配给所有客户端 (模拟全局底层微量通信，如DNS探测)
    # 这从物理层面防止了本地类概率绝对为0导致的数学崩溃
    chunks = np.array_split(noise_idx, num_clients)
    for i in range(num_clients):
        client_data_idx[i].extend(chunks[i].tolist())
        np.random.shuffle(client_data_idx[i])

    return client_data_idx


def make_global_long_tail(df, total_samples=10000, zipf_alpha=1.5):
    """使用 Zipf 定律构造全局长尾"""
    class_counts = df['label'].value_counts().sort_values(ascending=False)
    classes = class_counts.index.tolist()
    num_classes = len(classes)

    ranks = np.arange(1, num_classes + 1)
    weights = 1.0 / (ranks ** zipf_alpha)
    probabilities = weights / weights.sum()

    sampled_dfs = []
    for i, cls in enumerate(classes):
        expected_n = int(total_samples * probabilities[i])
        cls_df = df[df['label'] == cls]
        cls_available = len(cls_df)
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

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    test_df.to_csv("./dataset/global_test.csv", index=False)

    TARGET_TOTAL = 12000
    TOTAL_SAMPLES_TO_USE = min(TARGET_TOTAL, int(len(train_df) * 0.8))
    ZIPF_ALPHA = 1.5

    num_clients = 10
    alphas = [0.1]
    num_classes = len(le.classes_)

    train_df_rwth = make_global_long_tail(train_df, total_samples=TOTAL_SAMPLES_TO_USE, zipf_alpha=ZIPF_ALPHA)
    train_df_simple = train_df.sample(n=len(train_df_rwth), random_state=42)

    for alpha in alphas:
        splits = {
            "simple": (train_df_simple, simple_dirichlet_split(train_df_simple['label'].values, num_clients, alpha)),
            # 实验组：应用本文设计的真实流量混合划分！
            "rwth": (train_df_rwth, realistic_traffic_split(train_df_rwth['label'].values, num_clients, alpha, noise_ratio=0.03))
        }

        for split_type, (base_df, client_indices) in splits.items():
            save_dir = f"./dataset/{split_type}_alpha_{alpha}"
            os.makedirs(save_dir, exist_ok=True)

            for c in range(num_clients):
                client_df = base_df.iloc[client_indices[c]]
                client_df.to_csv(f"{save_dir}/client_{c}.csv", index=False)

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

    print(f"数据划分成功！模式: Simple组 vs RWTH混合真实组(Zipf={ZIPF_ALPHA})")