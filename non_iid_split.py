import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def simple_dirichlet_split(y, num_clients, alpha=0.5):
    """Dirichlet 划分"""
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


def realistic_traffic_split(y, num_clients, noise_ratio=0.15):
    num_samples = len(y)
    num_classes = len(np.unique(y))

    # 数量倾斜 (Lognormal)
    samples_per_client = np.random.lognormal(mean=3.0, sigma=1.2, size=num_clients)
    samples_per_client = (samples_per_client / samples_per_client.sum() * num_samples).astype(int)
    samples_per_client[np.argmax(samples_per_client)] += num_samples - samples_per_client.sum()
    indices_by_class = [np.where(y == i)[0].tolist() for i in range(num_classes)]
    for cls_list in indices_by_class: np.random.shuffle(cls_list)
    client_data_idx = [[] for _ in range(num_clients)]
    background_pool = []

    # 每个客户端主导类 (占 1 - noise_ratio)
    for c in range(num_clients):
        target_total = samples_per_client[c]
        target_main = int(target_total * (1.0 - noise_ratio))
        main_cls_1 = c % num_classes
        main_cls_2 = (c + 1) % num_classes
        take_1 = min(len(indices_by_class[main_cls_1]), int(target_main * 0.6))
        client_data_idx[c].extend(indices_by_class[main_cls_1][:take_1])
        indices_by_class[main_cls_1] = indices_by_class[main_cls_1][take_1:]
        take_2 = min(len(indices_by_class[main_cls_2]), target_main - take_1)
        client_data_idx[c].extend(indices_by_class[main_cls_2][:take_2])
        indices_by_class[main_cls_2] = indices_by_class[main_cls_2][take_2:]
    for cls in range(num_classes):
        background_pool.extend(indices_by_class[cls])
    np.random.shuffle(background_pool)
    current_idx = 0
    for c in range(num_clients):
        target_total = samples_per_client[c]
        current_len = len(client_data_idx[c])
        need = target_total - current_len
        if need > 0 and current_idx < len(background_pool):
            take_bg = min(need, len(background_pool) - current_idx)
            client_data_idx[c].extend(background_pool[current_idx : current_idx + take_bg])
            current_idx += take_bg
    while current_idx < len(background_pool):
        client_data_idx[np.random.randint(num_clients)].append(background_pool[current_idx])
        current_idx += 1
    return client_data_idx

def make_global_long_tail(df, total_samples=10000, zipf_alpha=1.5):
    """Zipf 定律构造长尾"""
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

    if pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].astype(int)

        unique_labels = sorted(df['label'].unique())
        label_remap = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

        df['label'] = df['label'].map(label_remap).astype(int)
        class_names = [str(old_label) for old_label in unique_labels]

    else:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'].astype(str))
        class_names = le.classes_.tolist()

    class_counts = df['label'].value_counts()
    test_samples_per_class = min(1500, max(class_counts.min() // 2, 50))

    test_dfs, train_dfs = [], []
    for cls in class_counts.index:
        cls_df = df[df['label'] == cls]
        # 针对极度稀有类做安全处理
        n_test = min(test_samples_per_class, len(cls_df) - 5)
        test_df_cls = cls_df.sample(n=max(n_test, 1), random_state=42)
        test_dfs.append(test_df_cls)
        train_dfs.append(cls_df.drop(test_df_cls.index))

    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=42)
    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=42)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    test_df.to_csv("./dataset/global_test.csv", index=False)

    print(f"测试集构建完成: 共 {len(test_df)} 条样本")

    # RWTH 组
    TARGET_TOTAL = 250000
    TOTAL_SAMPLES_TO_USE = min(TARGET_TOTAL, int(len(train_df) * 0.9))
    ZIPF_ALPHA = 1.25

    num_clients = 10
    alphas = [0.1,0.3,0.5,0.7]
    num_classes = len(class_names)

    # Simple 组
    TARGET_SIMPLE = 250000
    train_df_simple = train_df.sample(n=min(TARGET_SIMPLE, len(train_df)), random_state=42)

    # 构造 RWTH 长尾分布
    train_df_rwth = make_global_long_tail(train_df, total_samples=TOTAL_SAMPLES_TO_USE, zipf_alpha=ZIPF_ALPHA)

    print(f"Simple组采样: {len(train_df_simple)} | RWTH组采样: {len(train_df_rwth)}")

    for alpha in alphas:
        splits = {
            "simple": (train_df_simple, simple_dirichlet_split(train_df_simple['label'].values, num_clients, alpha)),
            "rwth": (
            train_df_rwth, realistic_traffic_split(train_df_rwth['label'].values, num_clients, noise_ratio=0.05))
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
            "classes": class_names,
            "total_training_samples": len(train_df_rwth)
        }, f)

    print(f"数据划分成功！")