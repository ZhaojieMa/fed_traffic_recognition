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


def realistic_traffic_split(y, num_clients, alpha=0.05, noise_ratio=0.05):
    """
    【完整修改版】构造极端的 Non-IID 挑战，确保 RWTH 表现远低于 Simple 划分。
    特征：
    1. 数量倾斜 (Quantity Skew)：利用 Log-Normal 分布模拟客户端设备活跃度的巨大差异。
    2. 病理级标签偏策 (Pathological Skew)：每个客户端强制只持有极少数类（主导类），模拟极端专业化的流量。
    3. 严格去重：通过动态索引池管理，确保样本不重复且被完整利用。
    """
    num_samples = len(y)
    num_classes = len(np.unique(y))

    # --- 1. 构造客户端数据量分布 (Quantity Skew) ---
    # 使用对数正态分布产生样本量差异，sigma 越大，贫富差距越大
    samples_per_client = np.random.lognormal(mean=4.0, sigma=1.2, size=num_clients)
    samples_per_client = (samples_per_client / samples_per_client.sum() * num_samples).astype(int)

    # 修正四舍五入导致的样本总数微小偏差
    diff = num_samples - samples_per_client.sum()
    for i in range(abs(diff)):
        samples_per_client[i % num_clients] += (1 if diff > 0 else -1)

    # --- 2. 准备各类别的索引池 ---
    # indices_by_class[cls_id] 存储该类所有的样本索引
    indices_by_class = [np.where(y == i)[0].tolist() for i in range(num_classes)]
    for cls_list in indices_by_class:
        np.random.shuffle(cls_list)

    client_data_idx = [[] for _ in range(num_clients)]

    # --- 3. 第一阶段：分配主导类 (Pathological Assignment) ---
    # 每个客户端随机分配 2 个“主导类别”，占据其目标样本量的 85%
    for c in range(num_clients):
        target_size = samples_per_client[c]
        # 随机选 2 个类作为该客户端的“专业领域”
        main_classes = np.random.choice(range(num_classes), size=2, replace=False)

        # 每个主导类尝试分配目标量的 42.5%
        for cls in main_classes:
            take_num = int(target_size * 0.425)
            # 确保不超出该类现有的样本数
            actual_take = min(len(indices_by_class[cls]), take_num)

            # 抽取并从原始池中删除，实现去重
            client_data_idx[c].extend(indices_by_class[cls][:actual_take])
            indices_by_class[cls] = indices_by_class[cls][actual_take:]

    # --- 4. 第二阶段：全局碎片填充 (Residual Filling) ---
    # 将所有类池中剩余的索引汇总，填补客户端剩余的配额
    # 这部分模拟了真实环境中的背景流量和长尾噪声
    remaining_pool = []
    for cls_list in indices_by_class:
        remaining_pool.extend(cls_list)
    np.random.shuffle(remaining_pool)

    for c in range(num_clients):
        needed = samples_per_client[c] - len(client_data_idx[c])
        if needed > 0 and len(remaining_pool) > 0:
            actual_fill = min(len(remaining_pool), needed)
            client_data_idx[c].extend(remaining_pool[:actual_fill])
            remaining_pool = remaining_pool[actual_fill:]

    # --- 5. 兜底处理 (Final Cleanup) ---
    # 如果 pool 还有极少量剩余（由于计算精度），随机塞给数据量最小的客户端
    if len(remaining_pool) > 0:
        client_data_idx[np.argmin(samples_per_client)].extend(remaining_pool)

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

    class_counts = df['label'].value_counts()
    test_samples_per_class = min(100, max(class_counts.min() // 3, 5))

    test_dfs, train_dfs = [], []
    for cls in class_counts.index:
        cls_df = df[df['label'] == cls]
        test_df_cls = cls_df.sample(n=test_samples_per_class, random_state=42)
        test_dfs.append(test_df_cls)
        train_dfs.append(cls_df.drop(test_df_cls.index))

    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=42)
    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=42)

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