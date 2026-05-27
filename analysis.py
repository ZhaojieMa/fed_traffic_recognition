import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = "./results"


def load_data():
    with open("./results/metrics.json", "r", encoding='utf-8') as f:
        return json.load(f)


def alpha_dir_name(alpha):
    return f"alpha_{str(alpha).replace('.', 'p')}"


def make_output_dir(alpha):
    out_dir = os.path.join(RESULTS_DIR, alpha_dir_name(alpha))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def make_all_alpha_dir():
    out_dir = os.path.join(RESULTS_DIR, "all_alpha")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_class_matrix(split_type, alpha="0.5", num_clients=10):
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


def plot_heatmap(alpha="0.5", output_dir=RESULTS_DIR):
    mat_simple = get_class_matrix("simple", alpha)
    mat_rwth = get_class_matrix("rwth", alpha)

    # 【核心修复】：取消按行归一化，改用 log1p 展现真实的数量级分布！
    # 这样能一目了然地看到 RWTH 中哪些是海量数据的“核心节点”，哪些是几乎空载的“边缘节点”
    mat_simple_log = np.log1p(mat_simple)
    mat_rwth_log = np.log1p(mat_rwth)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 统一全局刻度，凸显对比
    vmin = 0
    vmax = max(mat_simple_log.max(), mat_rwth_log.max())
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

    # 图1：对照组
    sns.heatmap(mat_simple_log, ax=axes[0], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Log(样本数 + 1)'})
    axes[0].set_xlabel("类别 (Class ID)", fontsize=12)
    axes[0].set_ylabel("客户端 (Client ID)", fontsize=12)

    # 图2：实验组
    sns.heatmap(mat_rwth_log, ax=axes[1], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Log(样本数 + 1)'})
    axes[1].set_xlabel("类别 (Class ID)", fontsize=12)
    axes[1].set_ylabel("客户端 (Client ID)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot1_heatmap_alpha_{alpha}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_histogram_and_coverage(alpha="0.5", output_dir=RESULTS_DIR):
    """数量分布与覆盖率"""
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
    axes[0].bar(x + 0.2, cls_per_client_p, 0.4, label="RWTH", color='coral')
    axes[0].set_xlabel("客户端 ID")
    axes[0].set_ylabel("类别数量")
    axes[0].set_xticks(x)
    axes[0].legend()

    # 覆盖曲线
    axes[1].plot(client_per_cls_s, marker='o', linestyle='--', color='gray', label="单纯 Dirichlet")
    axes[1].plot(client_per_cls_p, marker='s', linewidth=2, color='coral', label="RWTH")
    axes[1].set_xlabel("类别 ID")
    axes[1].set_ylabel("包含该类的客户端数量")
    axes[1].set_ylim(0, 11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot2_distribution_coverage_alpha_{alpha}.png"), dpi=300)
    plt.close()


def plot_convergence(data, alpha="0.5", output_dir=RESULTS_DIR):
    """
    曲线收敛对比图
    """
    # 提取 Simple 环境数据
    hist_simple_avg = data[alpha]["simple"]["FedAvg"]["hist"]
    hist_simple_prox = data[alpha]["simple"]["FedProx"]["hist"]
    hist_simple_prop = data[alpha]["simple"]["Proposed"]["hist"]

    # 提取 RWTH 环境数据
    hist_rwth_avg = data[alpha]["rwth"]["FedAvg"]["hist"]
    hist_rwth_prox = data[alpha]["rwth"]["FedProx"]["hist"]
    hist_rwth_prop = data[alpha]["rwth"]["Proposed"]["hist"]

    plt.figure(figsize=(12, 7))
    rounds = range(1, len(hist_simple_avg) + 1)

    # ================= 1. 绘制 Simple 组 ================
    plt.plot(rounds, hist_simple_avg, ':', color='#3498db', linewidth=2.5, alpha=0.6, label="FedAvg (单纯 Dirichlet)")
    plt.plot(rounds, hist_simple_prox, ':', color='#2ecc71', linewidth=2.5, alpha=0.6, label="FedProx (单纯 Dirichlet)")
    plt.plot(rounds, hist_simple_prop, ':', color='#e74c3c', linewidth=2.5, alpha=0.6,
             label="FedLC-Ada (单纯 Dirichlet)")

    # ================= 2. 绘制 RWTH 组 =================
    plt.plot(rounds, hist_rwth_avg, '-', color='#2980b9', linewidth=2, label="FedAvg (极限 RWTH)")
    plt.plot(rounds, hist_rwth_prox, '-', color='#27ae60', linewidth=2, label="FedProx (极限 RWTH)")
    plt.plot(rounds, hist_rwth_prop, '-', color='#c0392b', linewidth=3.5, label="FedLC-Ada (极限 RWTH)")

    plt.xlabel("通信轮数 (Rounds)", fontsize=12)
    plt.ylabel("全局测试集准确率 (Accuracy)", fontsize=12)

    # 使用双列图例，方便左右对比
    plt.legend(loc='lower right', ncol=2, frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot3_convergence_alpha_{alpha}.png"), dpi=300)
    plt.close()


def plot_classic_bar(data, alpha="0.5", output_dir=RESULTS_DIR):
    """柱状图"""
    methods = ["Local", "FedAvg", "FedProx", "FedLC-Ada", "Centralized"]
    keys = ["Local", "FedAvg", "FedProx", "Proposed", "Centralized"]

    accs = [data[alpha]["rwth"][k]["acc"] for k in keys]
    f1s = [data[alpha]["rwth"][k]["f1"] for k in keys]

    plt.figure(figsize=(11, 6))
    x = np.arange(len(methods))
    plt.bar(x - 0.2, accs, 0.4, label='准确率 (Accuracy)', color='#4c72b0', edgecolor='black')
    plt.bar(x + 0.2, f1s, 0.4, label='F1-score', color='#dd8452', edgecolor='black')

    for i, (a, f) in enumerate(zip(accs, f1s)):
        plt.text(i - 0.2, a + 0.01, f'{a:.4f}', ha='center', va='bottom')
        plt.text(i + 0.2, f + 0.01, f'{f:.4f}', ha='center', va='bottom')

    plt.xticks(x, methods, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"plot4_accuracy_f1_bar_alpha_{alpha}.png"), dpi=300)
    plt.close()


def plot_degradation_and_advantage(data, alpha="0.5", output_dir=RESULTS_DIR):
    """
    性能断崖验证与抗性对比
    """
    methods = ["FedAvg", "FedProx", "Proposed (本文方法)"]
    keys = ["FedAvg", "FedProx", "Proposed"]

    # 提取两种环境下的准确率
    acc_simple = [data[alpha]["simple"][k]["acc"] for k in keys]
    acc_rwth = [data[alpha]["rwth"][k]["acc"] for k in keys]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, acc_simple, width, label='Dirichlet', color='#95a5a6',
                    edgecolor='black')
    rects2 = ax.bar(x + width / 2, acc_rwth, width, label='RWTH 划分', color='#e74c3c', edgecolor='black')

    # 计算并标注性能下降幅度
    for i in range(len(methods)):
        drop_val = acc_simple[i] - acc_rwth[i]
        drop_pct = (drop_val / acc_simple[i]) * 100 if acc_simple[i] > 0 else 0

        # 在高柱子上标数值
        ax.text(x[i] - width / 2, acc_simple[i] + 0.01, f'{acc_simple[i]:.3f}', ha='center', va='bottom', fontsize=10)
        # 在矮柱子上标数值
        ax.text(x[i] + width / 2, acc_rwth[i] + 0.01, f'{acc_rwth[i]:.3f}', ha='center', va='bottom', fontsize=10)

        # 标注下降百分比
        font_weight = 'bold' if keys[i] == "Proposed" else 'normal'
        color = 'darkgreen' if keys[i] == "Proposed" else 'darkred'

        # 在柱子中间画下降箭头和文字
        ax.text(x[i] + width / 2, acc_rwth[i] + 0.05, f'↓{drop_pct:.1f}%',
                ha='center', va='bottom', color=color, fontweight=font_weight, fontsize=11)

    ax.set_ylabel('全局测试集准确率 (Accuracy)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot5_degradation_alpha_{alpha}.png"), dpi=300)
    plt.close()


def plot_ablation_study(data, alpha="0.5", output_dir=RESULTS_DIR):
    """
    消融实验
    """
    rwth_data = data[alpha]["simple"]

    plt.figure(figsize=(12, 7))
    i = 1
    while i <= 2:
        # 定义各组的颜色和线型，突出 Proposed
        if i == 1:
            configs = {
                "FedProx": {"color": "#3498db", "ls": "-.", "lw": 2, "label": "1. FedProx"},
                "DP": {"color": "#f39c12", "ls": "-", "lw": 2.5, "label": "2. DP"},
                "LA": {"color": "#2ecc71", "ls": "-", "lw": 2.5, "label": "3. LA"},
                "FL": {"color": "#8e44ad", "ls": "-", "lw": 2.5, "label": "4. FL"},
                "Proposed": {"color": "#c0392b", "ls": "-", "lw": 4, "label": "5. FedLC-Ada"}
            }
        else:
            configs = {
                "FedProx": {"color": "#3498db", "ls": "-.", "lw": 2, "label": "1. FedProx"},
                "LA+FL": {"color": "#f39c12", "ls": "-", "lw": 2.5, "label": "2. LA+FL"},
                "LA+DP": {"color": "#2ecc71", "ls": "-", "lw": 2.5, "label": "3. LA+DP"},
                "DP+FL": {"color": "#8e44ad", "ls": "-", "lw": 2.5, "label": "4. DP+FL"},
                "Proposed": {"color": "#c0392b", "ls": "-", "lw": 4, "label": "5. FedLC-Ada"}
            }
        rounds = None
        for method, conf in configs.items():
            if method in rwth_data:
                hist = rwth_data[method]["hist"]
                if rounds is None:
                    rounds = range(1, len(hist) + 1)
                plt.plot(rounds, hist, label=conf["label"], color=conf["color"],
                         linestyle=conf["ls"], linewidth=conf["lw"])

        plt.xlabel("通信轮数 (Rounds)", fontsize=12)
        plt.ylabel("全局测试集准确率 (Accuracy)", fontsize=12)
        plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if i == 1:
            plt.savefig(os.path.join(output_dir, f"plot6_ablation_one_alpha_{alpha}.png"), dpi=300)
        else:
            plt.savefig(os.path.join(output_dir, f"plot6_ablation_two_alpha_{alpha}.png"), dpi=300)
        plt.close()
        i = i + 1


def load_class_names():
    """从 meta.json 中读取类别名称"""
    with open("./dataset/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "classes" in meta:
        return [str(c) for c in meta["classes"]]

    return [str(i) for i in range(meta["num_classes"])]


def plot_per_class_f1_comparison(data, alpha="0.5", split_type="rwth", output_dir=RESULTS_DIR):
    """
    绘制 FedAvg、FedProx、FedLC-Ada 在每一类上的 F1 对比。
    split_type 可选：
        "simple"：单纯 Dirichlet 组
        "rwth"：RWTH 高异构长尾组
    """
    methods = ["FedAvg", "FedProx", "Proposed"]
    labels = {
        "FedAvg": "FedAvg",
        "FedProx": "FedProx",
        "Proposed": "FedLC-Ada"
    }

    colors = {
        "FedAvg": "#20CBEF",
        "FedProx": "#BCEFA0",
        "Proposed": "#F08A8C"
    }

    class_names = load_class_names()
    num_classes = len(class_names)

    f1_values = []

    for method in methods:
        method_result = data[alpha][split_type][method]

        if "per_class_f1" not in method_result:
            raise KeyError(
                f"metrics.json 中缺少 {alpha}-{split_type}-{method} 的 per_class_f1。"
                f"请先按要求修改 fed_train.py 并重新跑实验。"
            )

        f1_values.append(method_result["per_class_f1"])

    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(max(12, num_classes * 0.75), 6))

    for i, method in enumerate(methods):
        plt.bar(
            x + (i - 1) * width,
            f1_values[i],
            width,
            label=labels[method],
            color=colors[method]
        )

    plt.xlabel("类别", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    plt.xticks(x, class_names, rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=True)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"plot7_per_class_f1_{split_type}_alpha_{alpha}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_macro_f1_summary(data, split_type="rwth", output_dir=None, alphas=None):
    """
    绘制四种 alpha 值下的 Macro-F1 对比图。
    """
    if output_dir is None:
        output_dir = make_all_alpha_dir()
    os.makedirs(output_dir, exist_ok=True)

    if alphas is None:
        alphas = sorted(data.keys(), key=lambda x: float(x))
    else:
        alphas = [str(alpha) for alpha in alphas]

    method_keys = ["Local", "FedAvg", "FedProx", "Proposed", "Centralized"]
    method_labels = ["Local", "FedAvg", "FedProx", "FedLC-Ada", "Centralized"]
    colors = ['#C3D0DD', '#20CBEF', '#BCEFA0', '#F08A8C', '#54E8BA']

    f1_values = {method: [] for method in method_keys}
    for alpha in alphas:
        split_data = data[alpha][split_type]
        for method in method_keys:
            f1_values[method].append(float(split_data[method]["f1"]))

    x = np.arange(len(alphas))
    width = min(0.15, 0.8 / len(method_keys))

    plt.figure(figsize=(10, 6))

    for i, (method, label) in enumerate(zip(method_keys, method_labels)):
        offset = (i - (len(method_keys) - 1) / 2) * width
        values = f1_values[method]
        plt.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors[i]
        )

        for j, value in enumerate(values):
            plt.text(
                x[j] + offset,
                value + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(x, alphas, fontsize=11)
    plt.xlabel(r"$\alpha$", fontsize=12)
    plt.ylim(0, 1.10)
    plt.ylabel("Macro-F1", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"plot_macro_f1_summary_{split_type}_all_alpha.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def get_rounds_to_baseline_from_hist(method_result, baseline_acc):
    hist = method_result.get("hist", [])
    hist_rounds = method_result.get("hist_rounds", [])

    if hist_rounds and len(hist_rounds) == len(hist):
        pairs = zip(hist_rounds, hist)
    else:
        pairs = enumerate(hist, start=1)

    for r, acc in pairs:
        if float(acc) + 1e-12 >= baseline_acc:
            return int(r)

    return None


def plot_communication_efficiency(data, output_dir=None):
    """
    通信效率对比图。
    """
    if output_dir is None:
        output_dir = make_all_alpha_dir()

    alphas = sorted(data.keys(), key=lambda x: float(x))

    methods = ["FedAvg", "FedProx", "Proposed"]
    labels = {
        "FedAvg": "FedAvg",
        "FedProx": "FedProx",
        "Proposed": "FedLC-Ada"
    }

    markers = {
        "FedAvg": "D",
        "FedProx": "s",
        "Proposed": "^"
    }

    colors = {
        "FedAvg": "#7f83c6",
        "FedProx": "#2dbb7f",
        "Proposed": "#24b7c9"
    }

    x = [float(a) for a in alphas]
    y_values = {m: [] for m in methods}

    fixed_baseline_round = 80

    for alpha in alphas:
        rwth = data[alpha]["rwth"]

        # FedAvg 第80轮准确率作为基准
        fedavg_hist = rwth["FedAvg"]["hist"]
        if len(fedavg_hist) >= fixed_baseline_round:
            baseline_acc = fedavg_hist[fixed_baseline_round - 1]
        else:
            baseline_acc = rwth["FedAvg"]["acc"]

        for m in methods:
            if m == "FedAvg":
                y_values[m].append(fixed_baseline_round)
            else:
                round_num = get_rounds_to_baseline_from_hist(rwth[m], baseline_acc)
                y_values[m].append(np.nan if round_num is None else round_num)

    plt.figure(figsize=(8.5, 4.8))

    for m in methods:
        plt.plot(
            x,
            y_values[m],
            marker=markers[m],
            markersize=7,
            linewidth=2.0,
            color=colors[m],
            label=labels[m]
        )

    plt.xlabel(r"$\alpha$", fontsize=13)
    plt.ylabel("通信轮次", fontsize=13)
    plt.xticks(x, [str(a) for a in alphas], fontsize=11)

    valid_y = [
        v for values in y_values.values()
        for v in values
        if v is not None and not np.isnan(v)
    ]
    plt.ylim(0, max(110, max(valid_y) + 10) if valid_y else 120)

    plt.grid(axis="y", linestyle="--", alpha=0.35)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_communication_efficiency_rwth_all_alpha.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def generate_all_plots_for_alpha(data, alpha):
    """为单个 alpha 生成全部图表，保存到 results/alpha_xxx/"""
    output_dir = make_output_dir(alpha)

    plot_classic_bar(data, alpha, output_dir)
    plot_heatmap(alpha, output_dir)
    plot_histogram_and_coverage(alpha, output_dir)
    plot_convergence(data, alpha, output_dir)
    plot_degradation_and_advantage(data, alpha, output_dir)
    plot_ablation_study(data, alpha, output_dir)
    plot_per_class_f1_comparison(data, alpha, "rwth", output_dir)

    print(f"[完成] α={alpha} 的图表已保存到: {output_dir}")


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    data = load_data()
    alphas = sorted(data.keys(), key=lambda x: float(x))
    print(f"检测到 alpha 列表: {alphas}")

    for alpha in alphas:
        generate_all_plots_for_alpha(data, alpha)

    all_alpha_dir = make_all_alpha_dir()
    plot_communication_efficiency(data, all_alpha_dir)
    plot_macro_f1_summary(data, "rwth", all_alpha_dir)

    print("\n所有图表生成完毕，请检查 ./results 文件夹！")
