import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    with open("./results/metrics.json", "r", encoding='utf-8') as f:
        return json.load(f)


def get_class_matrix(split_type, alpha="0.1", num_clients=10, num_classes=8):
    """读取真实的CSV获取客户端类分布矩阵"""
    with open("./dataset/meta.json", "r") as f:
        num_classes = json.load(f)["num_classes"]

    matrix = np.zeros((num_clients, num_classes))
    for c in range(num_clients):
        df = pd.read_csv(f"./dataset/{split_type}_alpha_{alpha}/client_{c}.csv")
        counts = df['label'].value_counts()
        for k, v in counts.items():
            matrix[c, int(k)] = v
    return matrix


def plot_heatmap(alpha="0.1"):
    mat_simple = get_class_matrix("simple", alpha)
    mat_prop = get_class_matrix("rwth", alpha) # 修改此处

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax_val = max(mat_simple.max(), mat_prop.max()) * 0.8

    sns.heatmap(mat_simple.T, ax=axes[0], cmap="YlOrRd", vmax=vmax_val, cbar_kws={'label': '样本数量'})
    axes[0].set_title(f"单纯 Dirichlet (α={alpha})\n分布随机，整体数量相对均衡", fontweight='bold')
    axes[0].set_xlabel("客户端 ID")
    axes[0].set_ylabel("类别 ID")

    sns.heatmap(mat_prop.T, ax=axes[1], cmap="YlOrRd", vmax=vmax_val, cbar_kws={'label': '样本数量'})
    axes[1].set_title(f"本文 RWTH 划分 (α={alpha})\n全局长尾 + 极端稀缺孤岛 (最严酷测试)", fontweight='bold')
    axes[1].set_xlabel("客户端 ID")
    axes[1].set_ylabel("类别 ID")

    plt.tight_layout()
    plt.savefig("./results/plot1_heatmap.png", dpi=300)


def plot_histogram_and_coverage(alpha="0.1"):
    """图2 & 图3：数量分布与覆盖率"""
    mat_simple = get_class_matrix("simple", alpha)
    mat_prop = get_class_matrix("rwth", alpha)

    # 客户端包含的类别数
    cls_per_client_s = (mat_simple > 0).sum(axis=1)
    cls_per_client_p = (mat_prop > 0).sum(axis=1)

    # 每个类别覆盖的客户端数
    client_per_cls_s = (mat_simple > 0).sum(axis=0)
    client_per_cls_p = (mat_prop > 0).sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    x = np.arange(10)
    axes[0].bar(x - 0.2, cls_per_client_s, 0.4, label="单纯 Dirichlet", color='lightgray')
    axes[0].bar(x + 0.2, cls_per_client_p, 0.4, label="本文改进", color='coral')
    axes[0].set_title("图2：每个客户端拥有的类别数量对比", fontweight='bold')
    axes[0].set_xlabel("客户端 ID")
    axes[0].set_ylabel("类别数量")
    axes[0].set_xticks(x)
    axes[0].legend()

    # 覆盖曲线
    axes[1].plot(client_per_cls_s, marker='o', linestyle='--', color='gray', label="单纯 Dirichlet")
    axes[1].plot(client_per_cls_p, marker='s', linewidth=2, color='coral', label="本文改进")
    axes[1].set_title("图3：类别在全局客户端中的覆盖率", fontweight='bold')
    axes[1].set_xlabel("类别 ID")
    axes[1].set_ylabel("包含该类的客户端数量")
    axes[1].set_ylim(0, 11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("./results/plot2_3_distribution.png", dpi=300)


def plot_convergence(data, alpha="0.1"):
    hist_simple = data[alpha]["simple"]["FedAvg"]["hist"]
    hist_prop_avg = data[alpha]["rwth"]["FedAvg"]["hist"] # 修改此处
    hist_prop_prox = data[alpha]["rwth"]["FedProx"]["hist"] # 修改此处
    hist_prop_alg = data[alpha]["rwth"]["Proposed"]["hist"] # 修改此处

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(hist_simple) + 1)

    plt.plot(rounds, hist_simple, ':', color='#7f8c8d', linewidth=2, label="FedAvg (单纯 Dirichlet数据)")
    plt.plot(rounds, hist_prop_avg, '--', color='#3498db', linewidth=2, alpha=0.8, label="FedAvg")
    plt.plot(rounds, hist_prop_prox, '-.', color='#2ecc71', linewidth=2.5, alpha=0.9, label="FedProx")
    plt.plot(rounds, hist_prop_alg, '-', color='#e74c3c', linewidth=3.5, label="本文方法 FedLC-Ada")

    plt.title(f"图4：RWTH 极限挑战下的收敛曲线对比 (α={alpha})", fontsize=15, fontweight='bold')
    plt.xlabel("通信轮数 (Rounds)", fontsize=12)
    plt.ylabel("测试集准确率 (Accuracy)", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("./results/plot4_convergence.png", dpi=300)


def plot_classic_bar(data, alpha="0.1"):
    """原版柱状图修复"""
    methods = ["本地独立", "FedAvg", "FedProx", "本文方法(FedLC-Ada)", "集中式(上限)"]
    keys = ["Local", "FedAvg", "FedProx", "Proposed", "Centralized"]

    accs = [data[alpha]["rwth"][k]["acc"] for k in keys]
    f1s = [data[alpha]["rwth"][k]["f1"] for k in keys]

    plt.figure(figsize=(11, 6))
    x = np.arange(len(methods))
    plt.bar(x - 0.2, accs, 0.4, label='准确率 (Accuracy)', color='#4c72b0', edgecolor='black')
    plt.bar(x + 0.2, f1s, 0.4, label='F1-score', color='#dd8452', edgecolor='black')

    for i, (a, f) in enumerate(zip(accs, f1s)):
        plt.text(i - 0.2, a + 0.01, f'{a:.3f}', ha='center', va='bottom')
        plt.text(i + 0.2, f + 0.01, f'{f:.3f}', ha='center', va='bottom')

    plt.xticks(x, methods, fontsize=11)
    plt.ylim(0, 1.1)
    plt.title(f"RWTH 真实长尾异构场景下各方法性能对比 (α={alpha})", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig("./results/accuracy_comparison.png", dpi=300)


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    data = load_data()

    # 1. 经典性能对比柱状图
    plot_classic_bar(data, alpha="0.1")

    # 2. 生成四大证明图表（强烈建议用 alpha=0.1 展示，对比最明显）
    target_alpha = "0.1"
    plot_heatmap(target_alpha)
    plot_histogram_and_coverage(target_alpha)
    plot_convergence(data, target_alpha)

    print("\n所有图表生成完毕，请检查 ./results 文件夹！")