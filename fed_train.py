import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import flwr as fl
from collections import OrderedDict
from model import TrafficResNet, fedprox_loss, fedlc_ada_loss


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./dataset/meta.json", "r") as f: META = json.load(f)
INPUT_DIM = META["input_dim"]
NUM_CLASSES = META["num_classes"]

NUM_CLIENTS = 10
EPOCHS_PER_ROUND = 5
TOTAL_ROUNDS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MU_PROX = 0.001
MU_ADA = 0.001
DIRICHLET_ALPHAS = [0.1,0.3,0.5,0.7]

# 通信效率统计：以 FedAvg 第 100 轮 Accuracy 作为基准
COMM_BASELINE_METHOD = "FedAvg"
COMM_BASELINE_ROUND = 100
COMM_METHODS = ["FedAvg", "FedProx", "Proposed"]

def load_global_test():
    df = pd.read_csv("./dataset/global_test.csv")
    X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(df['label'].values, dtype=torch.long).to(DEVICE)
    return X, y


GLOBAL_X_TEST, GLOBAL_Y_TEST = load_global_test()

def get_post_round_metric_pairs(history, metric_name):
    """
    从 Flower 的 History 中提取中心化评估指标。
    只保留 server_round > 0 的通信轮次，避免把第 0 轮初始评估计入通信轮次。
    """
    pairs = history.metrics_centralized.get(metric_name, [])
    pairs = [(int(server_round), float(value)) for server_round, value in pairs]
    post_round_pairs = [(server_round, value) for server_round, value in pairs if server_round > 0]
    return post_round_pairs if post_round_pairs else pairs

def robust_tail_mean(values, tail_n=14, trim_n=2):
    """
    稳健收敛结果：
    取最后 tail_n 个值，去掉 trim_n 个最大值和 trim_n 个最小值后求平均。
    若可用值不足，则自动退化为普通平均。
    """
    values = [float(v) for v in values if v is not None]

    if len(values) == 0:
        return None

    tail_values = values[-tail_n:]

    # 如果数量不足以去掉最大/最小，则直接平均
    if len(tail_values) <= 2 * trim_n:
        return float(np.mean(tail_values))

    sorted_values = sorted(tail_values)
    trimmed_values = sorted_values[trim_n:-trim_n]

    return float(np.mean(trimmed_values))

def result_round_pairs(result):
    """
    将 summary 中保存的 hist 与 hist_rounds 还原为 (round, accuracy) 对。
    若旧结果中没有 hist_rounds，则默认 hist 第一个点对应第 1 轮。
    """
    hist = result.get("hist", [])
    hist_rounds = result.get("hist_rounds", [])

    if hist_rounds and len(hist_rounds) == len(hist):
        return [(int(r), float(v)) for r, v in zip(hist_rounds, hist)]

    return [(i, float(v)) for i, v in enumerate(hist, start=1)]


def accuracy_at_round(round_acc_pairs, target_round):
    """
    获取指定通信轮次的准确率。
    若不存在精确轮次，则使用小于等于 target_round 的最近一次评估结果。
    """
    if not round_acc_pairs:
        return None

    round_acc_pairs = sorted(round_acc_pairs, key=lambda x: x[0])

    for server_round, acc in round_acc_pairs:
        if server_round == target_round:
            return float(acc)

    previous = [(r, a) for r, a in round_acc_pairs if r <= target_round]
    if previous:
        return float(previous[-1][1])

    return None


def first_round_to_accuracy(round_acc_pairs, target_acc):
    """
    统计首次达到或超过 target_acc 的通信轮次。
    若训练结束仍未达到，则返回 None。
    """
    for server_round, acc in sorted(round_acc_pairs, key=lambda x: x[0]):
        if server_round > 0 and acc + 1e-12 >= target_acc:
            return int(server_round)
    return None


def add_rwth_communication_efficiency(rwth_results):
    """
    在 rwth_results 中直接加入通信效率统计结果。
    基准：RWTH 场景下 FedAvg 第 COMM_BASELINE_ROUND 轮 Accuracy。
    统计：FedAvg、FedProx、Proposed 首次达到该 Accuracy 所需通信轮次。
    """
    fedavg_result = rwth_results.get(COMM_BASELINE_METHOD)
    if fedavg_result is None:
        print("[通信效率] 未找到 FedAvg 结果，跳过通信效率统计。")
        return

    fedavg_pairs = result_round_pairs(fedavg_result)
    baseline_acc = accuracy_at_round(fedavg_pairs, COMM_BASELINE_ROUND)

    if baseline_acc is None:
        print(f"[通信效率] 未找到 FedAvg 第 {COMM_BASELINE_ROUND} 轮准确率，跳过通信效率统计。")
        return

    rounds_to_baseline = {}

    for method in COMM_METHODS:
        method_result = rwth_results.get(method)
        if method_result is None:
            rounds_to_baseline[method] = None
            continue

        method_pairs = result_round_pairs(method_result)
        rounds_to_baseline[method] = first_round_to_accuracy(method_pairs, baseline_acc)

    rwth_results["communication_efficiency"] = {
        "baseline_method": COMM_BASELINE_METHOD,
        "baseline_round": COMM_BASELINE_ROUND,
        "baseline_accuracy": baseline_acc,
        "rounds_to_baseline_accuracy": rounds_to_baseline
    }

    print(f"\n[通信效率] RWTH 基准：FedAvg 第 {COMM_BASELINE_ROUND} 轮 Accuracy = {baseline_acc:.4f}")
    for method, round_num in rounds_to_baseline.items():
        display_name = "FedLC-Ada" if method == "Proposed" else method
        print(f"[通信效率] {display_name} 达到基准准确率所需通信轮次：{round_num}")

def load_client_data(client_id, alpha, split_type):
    data_path = f"./dataset/{split_type}_alpha_{alpha}/client_{client_id}.csv"
    dist_path = f"./dataset/{split_type}_alpha_{alpha}/client_{client_id}_dist.npy"
    df = pd.read_csv(data_path)
    X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(df['label'].values, dtype=torch.long).to(DEVICE)
    label_dist = torch.tensor(np.load(dist_path), dtype=torch.float32).to(DEVICE)
    return X, y, label_dist


class TrafficClient(fl.client.NumPyClient):
    def __init__(self, client_id, alpha, method, split_type):
        self.method = method
        self.split_type = split_type
        self.X_train, self.y_train, self.label_dist = load_client_data(client_id, alpha, split_type)
        self.model = TrafficResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("server_round", 1)

        global_model = None
        if self.method in ["FedProx", "Proposed", "DecoupledProx"]:
            global_model = TrafficResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
            global_model.load_state_dict(self.model.state_dict())
            global_model.eval()

        self.model.train()
        loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True)

        for _ in range(EPOCHS_PER_ROUND):
            for bx, by in loader:
                self.optimizer.zero_grad()
                outputs = self.model(bx)

                # 消融与主实验路由
                if self.method == "FedAvg":
                    loss = F.cross_entropy(outputs, by)
                elif self.method == "FedProx":
                    loss = fedprox_loss(outputs, by, self.model, global_model, mu=MU_PROX)
                elif self.method == "LA":
                    loss = fedlc_ada_loss(outputs, by, self.model, None, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=0, use_focal=False, use_decoupled_prox=False)
                elif self.method == "DP":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA, use_focal=False, use_la=False)
                elif self.method == "FL":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA, use_decoupled_prox=False, use_la=False)
                elif self.method == "LA+FL":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA, use_decoupled_prox=False)
                elif self.method == "LA+DP":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA, use_focal=False)
                elif self.method == "DP+FL":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA, use_la=False)
                elif self.method == "Proposed":
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist,
                                          current_round, TOTAL_ROUNDS, mu=MU_ADA)
                else : loss = F.cross_entropy(outputs, by)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.X_train), {}


def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        model = TrafficResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)

        params_dict = zip(model.state_dict().keys(), parameters)
        model.load_state_dict(
            OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict})
        )

        model.eval()

        with torch.no_grad():
            outputs = model(GLOBAL_X_TEST)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_y = GLOBAL_Y_TEST.cpu().numpy()

            acc = float(accuracy_score(true_y, preds))
            macro_f1 = float(f1_score(true_y, preds, average='macro', zero_division=0))

            # 每一类的 F1-score
            per_class_f1 = f1_score(
                true_y,
                preds,
                labels=np.arange(NUM_CLASSES),
                average=None,
                zero_division=0
            )

        metrics = {
            "accuracy": round(acc, 4),
            "f1": round(macro_f1, 4)
        }

        # Flower 的 metrics 字典需要是标量，因此逐类展开保存
        for cls_id, cls_f1 in enumerate(per_class_f1):
            metrics[f"f1_class_{cls_id}"] = round(float(cls_f1), 4)

        return 0.0, metrics

    return evaluate


def run_experiment(method, alpha, split_type="proposed"):
    print(f"\n[启动FL] 策略: {method} | 数据: {split_type} | α: {alpha}")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda r: {"server_round": r}
    )

    history = fl.simulation.start_simulation(
        client_fn=lambda cid: TrafficClient(int(cid), alpha, method, split_type).to_client(),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.2 if torch.cuda.is_available() else 0
        }
    )

    acc_pairs = get_post_round_metric_pairs(history, "accuracy")
    f1_pairs = get_post_round_metric_pairs(history, "f1")

    acc_rounds = [server_round for server_round, _ in acc_pairs]
    acc_hist = [val for _, val in acc_pairs]

    f1_rounds = [server_round for server_round, _ in f1_pairs]
    f1_hist = [val for _, val in f1_pairs]

    per_class_f1 = []
    per_class_f1_hist = {}
    per_class_f1_rounds = {}

    for cls_id in range(NUM_CLASSES):
        cls_pairs = get_post_round_metric_pairs(history, f"f1_class_{cls_id}")

        cls_rounds = [server_round for server_round, _ in cls_pairs]
        cls_hist = [val for _, val in cls_pairs]

        per_class_f1.append(
            robust_tail_mean(cls_hist, tail_n=14, trim_n=2)
        )

        per_class_f1_hist[str(cls_id)] = cls_hist
        per_class_f1_rounds[str(cls_id)] = cls_rounds

    # 原逻辑：直接取最后一轮
    last_acc = acc_hist[-1]
    last_f1 = f1_hist[-1]

    # 新逻辑：最后14轮去掉2个最大值和2个最小值后取平均
    final_acc = robust_tail_mean(acc_hist, tail_n=14, trim_n=2)
    final_f1 = robust_tail_mean(f1_hist, tail_n=14, trim_n=2)

    print(
        f"{method} | {split_type} | α={alpha} | "
        f"last_acc={last_acc:.4f}, final_acc={final_acc:.4f}, "
        f"last_f1={last_f1:.4f}, final_acc={final_f1:.4f}"
    )

    return (
        final_acc,
        final_f1,
        acc_hist,
        acc_rounds,
        f1_hist,
        f1_rounds,
        per_class_f1,
        per_class_f1_hist,
        per_class_f1_rounds
    )


def centralized_baseline(alpha, split_type="proposed"):
    print(f"\n[基准] 集中式上限 ({split_type} α={alpha})")
    all_x, all_y = [], []
    for i in range(NUM_CLIENTS):
        x, y, _ = load_client_data(i, alpha, split_type)
        all_x.append(x)
        all_y.append(y)

    loader = DataLoader(TensorDataset(torch.cat(all_x), torch.cat(all_y)), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)
    model = TrafficResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(50):
        for bx, by in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(bx), by).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(GLOBAL_X_TEST), dim=1).cpu().numpy()
        acc = accuracy_score(GLOBAL_Y_TEST.cpu().numpy(), preds)
        f1 = f1_score(GLOBAL_Y_TEST.cpu().numpy(), preds, average='macro')
    return float(acc), float(f1)


def local_only_training(alpha, split_type="proposed"):
    print(f"\n[基准] 本地独立训练 ({split_type} α={alpha})")
    all_client_accs, all_client_f1s = [], []
    for i in range(NUM_CLIENTS):
        X, y, _ = load_client_data(i, alpha, split_type)
        model = TrafficResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        model.train()
        for _ in range(TOTAL_ROUNDS * EPOCHS_PER_ROUND):
            for bx, by in loader:
                optimizer.zero_grad()
                F.cross_entropy(model(bx), by).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(GLOBAL_X_TEST), dim=1).cpu().numpy()
            true_y = GLOBAL_Y_TEST.cpu().numpy()
            all_client_accs.append(accuracy_score(true_y, preds))
            all_client_f1s.append(f1_score(true_y, preds, average='macro'))
    return float(np.mean(all_client_accs)), float(np.mean(all_client_f1s))


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)

    alphas = DIRICHLET_ALPHAS
    summary = {}

    for alpha in alphas:
        print(f"\n{'=' * 40}\n正在测试异构度 α = {alpha}\n{'=' * 40}")
        summary[str(alpha)] = {"simple": {}, "rwth": {}}

        # 1. Simple组
        # ablation_methods = ["FedProx" ,"DP","LA","FL","LA+FL","LA+DP","DP+FL","Proposed"]
        ablation_methods = ["FedAvg","FedProx","Proposed"]

        for m in ablation_methods:
            (
                acc, f1, hist, hist_rounds,
                f1_hist, f1_rounds,
                per_class_f1, per_class_f1_hist, per_class_f1_rounds
            ) = run_experiment(m, alpha, "simple")

            summary[str(alpha)]["simple"][m] = {
                "acc": acc,
                "f1": f1,
                "hist": hist,
                "hist_rounds": hist_rounds,
                "f1_hist": f1_hist,
                "f1_rounds": f1_rounds,
                "per_class_f1": per_class_f1,
                "per_class_f1_hist": per_class_f1_hist,
                "per_class_f1_rounds": per_class_f1_rounds
            }

        # 2. RWTH组
        l_acc, l_f1 = local_only_training(alpha, "rwth")
        c_acc, c_f1 = centralized_baseline(alpha, "rwth")

        (
            pa_acc, pa_f1, pa_hist, pa_rounds,
            pa_f1_hist, pa_f1_rounds,
            pa_per_class_f1, pa_per_class_f1_hist, pa_per_class_f1_rounds
        ) = run_experiment("FedAvg", alpha, "rwth")
        (
            pp_acc, pp_f1, pp_hist, pp_rounds,
            pp_f1_hist, pp_f1_rounds,
            pp_per_class_f1, pp_per_class_f1_hist, pp_per_class_f1_rounds
        ) = run_experiment("FedProx", alpha, "rwth")
        (
            po_acc, po_f1, po_hist, po_rounds,
            po_f1_hist, po_f1_rounds,
            po_per_class_f1, po_per_class_f1_hist, po_per_class_f1_rounds
        ) = run_experiment("Proposed", alpha, "rwth")

        summary[str(alpha)]["rwth"]["Local"] = {
            "acc": l_acc,
            "f1": l_f1
        }
        summary[str(alpha)]["rwth"]["Centralized"] = {
            "acc": c_acc,
            "f1": c_f1
        }
        summary[str(alpha)]["rwth"]["FedAvg"] = {
            "acc": pa_acc,
            "f1": pa_f1,
            "hist": pa_hist,
            "hist_rounds": pa_rounds,
            "f1_hist": pa_f1_hist,
            "f1_rounds": pa_f1_rounds,
            "per_class_f1": pa_per_class_f1,
            "per_class_f1_hist": pa_per_class_f1_hist,
            "per_class_f1_rounds": pa_per_class_f1_rounds
        }
        summary[str(alpha)]["rwth"]["FedProx"] = {
            "acc": pp_acc,
            "f1": pp_f1,
            "hist": pp_hist,
            "hist_rounds": pp_rounds,
            "f1_hist": pp_f1_hist,
            "f1_rounds": pp_f1_rounds,
            "per_class_f1": pp_per_class_f1,
            "per_class_f1_hist": pp_per_class_f1_hist,
            "per_class_f1_rounds": pp_per_class_f1_rounds
        }
        summary[str(alpha)]["rwth"]["Proposed"] = {
            "acc": po_acc,
            "f1": po_f1,
            "hist": po_hist,
            "hist_rounds": po_rounds,
            "f1_hist": po_f1_hist,
            "f1_rounds": po_f1_rounds,
            "per_class_f1": po_per_class_f1,
            "per_class_f1_hist": po_per_class_f1_hist,
            "per_class_f1_rounds": po_per_class_f1_rounds
        }

        add_rwth_communication_efficiency(summary[str(alpha)]["rwth"])

    with open("./results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n所有实验数据已保存，请运行 analysis.py")