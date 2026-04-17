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
from model import TrafficMLP, fedprox_loss, fedlc_ada_loss


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
TOTAL_ROUNDS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MU_PROX = 0.01
MU_ADA = 0.01


def load_global_test():
    df = pd.read_csv("./dataset/global_test.csv")
    X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(df['label'].values, dtype=torch.long).to(DEVICE)
    return X, y


GLOBAL_X_TEST, GLOBAL_Y_TEST = load_global_test()


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
        self.model = TrafficMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
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
        if self.method in ["FedProx", "Proposed"]:
            global_model = TrafficMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
            global_model.load_state_dict(self.model.state_dict())
            global_model.eval()

        self.model.train()
        loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for _ in range(EPOCHS_PER_ROUND):
            for bx, by in loader:
                self.optimizer.zero_grad()
                outputs = self.model(bx)
                if self.method == "Proposed":
                    # 核心修复：传入我们设定的 MU_ADA，而不是用原先的 0.5 锁死模型
                    loss = fedlc_ada_loss(outputs, by, self.model, global_model, self.label_dist, current_round, TOTAL_ROUNDS, mu=MU_ADA)
                elif self.method == "FedProx":
                    loss = fedprox_loss(outputs, by, self.model, global_model, mu=MU_PROX)
                else:
                    loss = F.cross_entropy(outputs, by)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.X_train), {}


def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        model = TrafficMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        model.load_state_dict(OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict}))
        model.eval()
        with torch.no_grad():
            outputs = model(GLOBAL_X_TEST)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_y = GLOBAL_Y_TEST.cpu().numpy()
            acc = float(accuracy_score(true_y, preds))
            f1 = float(f1_score(true_y, preds, average='macro'))
        return 0.0, {"accuracy": acc, "f1": f1}

    return evaluate


def run_experiment(method, alpha, split_type="proposed"):
    print(f"\n[启动FL] 策略: {method} | 数据: {split_type} | α: {alpha}")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(), on_fit_config_fn=lambda r: {"server_round": r}
    )
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: TrafficClient(int(cid), alpha, method, split_type).to_client(),
        num_clients=NUM_CLIENTS, config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy, client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0}
    )
    final_acc = history.metrics_centralized["accuracy"][-1][1]
    final_f1 = history.metrics_centralized["f1"][-1][1]
    acc_hist = [val for _, val in history.metrics_centralized["accuracy"]]
    return final_acc, final_f1, acc_hist


def centralized_baseline(alpha, split_type="proposed"):
    print(f"\n[基准] 集中式上限 ({split_type} α={alpha})")
    all_x, all_y = [], []
    for i in range(NUM_CLIENTS):
        x, y, _ = load_client_data(i, alpha, split_type)
        all_x.append(x)
        all_y.append(y)

    loader = DataLoader(TensorDataset(torch.cat(all_x), torch.cat(all_y)), batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    model = TrafficMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
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
        model = TrafficMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
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
    alphas = [0.1]
    summary = {}

    for alpha in alphas:
        print(f"\n{'=' * 40}\n正在测试异构度 α = {alpha}\n{'=' * 40}")
        summary[str(alpha)] = {"simple": {}, "rwth": {}}

        # 1. 运行 Simple (控制变量组：只有类别不均，没有长尾缺失)
        sa_acc, sa_f1, sa_hist = run_experiment("FedAvg", alpha, "simple")
        sp_acc, sp_f1, sp_hist = run_experiment("FedProx", alpha, "simple")
        so_acc, so_f1, so_hist = run_experiment("Proposed", alpha, "simple")
        summary[str(alpha)]["simple"]["FedAvg"] = {"acc": sa_acc, "f1": sa_f1, "hist": sa_hist}
        summary[str(alpha)]["simple"]["FedProx"] = {"acc": sp_acc, "f1": sp_f1, "hist": sp_hist}
        summary[str(alpha)]["simple"]["Proposed"] = {"acc": so_acc, "f1": so_f1, "hist": so_hist}

        # 2. 运行 RWTH (实验组：全局长尾 + 极端异构的双重地狱)
        l_acc, l_f1 = local_only_training(alpha, "rwth")
        c_acc, c_f1 = centralized_baseline(alpha, "rwth")

        pa_acc, pa_f1, pa_hist = run_experiment("FedAvg", alpha, "rwth")
        pp_acc, pp_f1, pp_hist = run_experiment("FedProx", alpha, "rwth")
        po_acc, po_f1, po_hist = run_experiment("Proposed", alpha, "rwth")

        summary[str(alpha)]["rwth"]["Local"] = {"acc": l_acc, "f1": l_f1}
        summary[str(alpha)]["rwth"]["Centralized"] = {"acc": c_acc, "f1": c_f1}
        summary[str(alpha)]["rwth"]["FedAvg"] = {"acc": pa_acc, "f1": pa_f1, "hist": pa_hist}
        summary[str(alpha)]["rwth"]["FedProx"] = {"acc": pp_acc, "f1": pp_f1, "hist": pp_hist}
        summary[str(alpha)]["rwth"]["Proposed"] = {"acc": po_acc, "f1": po_f1, "hist": po_hist}

    with open("./results/metrics.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print("\n所有实验数据已保存，请运行 analysis.py")