import nfstream
import pandas as pd
import numpy as np
import os
import json
import warnings
import re

warnings.filterwarnings('ignore')

# 数据集切换：
#   "iscx" -> D:/VPN-NonVPN-PCAPs-01
#   "ustc" -> D:/USTC-TFC2016
DATASET_NAME = "ustc"
# iscx不同任务
ISCX_TASK_MODE = 'application' # Encapsulation|category|application

DATASET_CONFIGS = {
    "iscx": {
        "target_dir": "D:/VPN-NonVPN-PCAPs-01",
        "description": "ISCX VPN-NonVPN"
    },
    "ustc": {
        "target_dir": "D:/USTC-TFC2016",
        "description": "USTC-TFC2016 10-class"
    }
}

UNUSED_FEATURES = [
    # 流标识、地址、端口、链路层信息
    'id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
    'dst_ip', 'dst_mac', 'dst_oui', 'dst_port',

    # NFStream 的 DPI/应用识别结果，训练应用分类时会造成标签泄漏
    'application_name', 'application_category_name', 'category_name',
    'application_confidence', 'application_is_guessed',
    'requested_server_name', 'client_fingerprint', 'server_fingerprint',
    'user_agent', 'content_type',

    # 内部、协议或环境相关字段
    'expiration_id', 'ip_version', 'protocol',
    'vlan_id', 'tunnel_id', 'entry_type',

    # SPLT 原始序列是 object/list 形态，不直接作为标量特征
    'splt_direction', 'splt_ps', 'splt_piat_ms'
]


def select_feature_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    unused_columns = set(UNUSED_FEATURES)
    return [col for col in numeric_columns if col not in unused_columns]


def normalize_feature_table(df):
    feature_cols = sorted([col for col in df.columns if col != 'label'])
    df = df[feature_cols + ['label']]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def get_dataset_config(dataset_name=DATASET_NAME):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}，可选值: {list(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[dataset_name]


def get_ustc_label(filename):
    name = filename.lower()

    if "bittorrent" in name:
        return "BitTorrent"
    elif "facetime" in name:
        return "Facetime"
    elif "ftp" in name:
        return "FTP"
    elif "gmail" in name:
        return "Gmail"
    elif "mysql" in name:
        return "MySQL"
    elif "outlook" in name:
        return "Outlook"
    elif "skype" in name:
        return "Skype"
    elif "smb" in name:
        return "SMB"
    elif "weibo" in name:
        return "Weibo"
    elif "worldofwarcraft" in name:
        return "WorldOfWarcraft"

    return "Unknown"


def get_iscx_label(filename):
    name = filename.lower()
    name = re.sub(r'\.(pcap|pcapng|csv)$', '', name)

    # ===== 1. VPN 标记 =====
    is_vpn = name.startswith('vpn_')
    pure_name = re.sub(r'^vpn_', '', name)

    # ===== 2. 去掉实验编号 =====
    pure_name = re.sub(r'\d+[a-z]?$', '', pure_name)
    pure_name = re.sub(r'_[ab]$', '', pure_name)

    # ===== 3. 二分类 =====
    if ISCX_TASK_MODE == 'Encapsulation':
        return "VPN" if is_vpn else "NonVPN"

    # ===== 4. 应用识别 =====
    app = None
    traffic_type = None

    # 识别应用
    if "facebook" in pure_name:
        app = "facebook"
    elif "skype" in pure_name:
        app = "skype"
    elif "hangout" in pure_name:
        app = "hangout"
    elif "gmail" in pure_name:
        app = "gmail"
    elif "email" in pure_name:
        app = "email"
    elif "youtube" in pure_name:
        app = "youtube"
    elif "netflix" in pure_name:
        app = "netflix"
    elif "spotify" in pure_name:
        app = "spotify"
    elif "vimeo" in pure_name:
        app = "vimeo"
    elif "aim" in pure_name:
        app = "aim"
    elif "icq" in pure_name:
        app = "icq"
    elif "voipbuster" in pure_name:
        app = "voipbuster"
    elif "scp" in pure_name:
        app = "scp"
    elif "sftp" in pure_name:
        app = "sftp"
    elif "ftp" in pure_name:
        app = "ftp"
    elif "bittorrent" in pure_name:
        app = "bittorrent"

    # ===== 5. 识别流量类型 =====
    if "audio" in pure_name:
        traffic_type = "audio"
    elif "video" in pure_name:
        traffic_type = "video"
    elif "chat" in pure_name:
        traffic_type = "chat"
    elif "file" in pure_name or "scp" in pure_name or "ftp" in pure_name:
        traffic_type = "file"
    else:
        traffic_type = "default"

    # ===== 6. 16分类 =====
    if ISCX_TASK_MODE == 'application':

        app_map = {
            "skype": "Skype",
            "icq": "ICQ",
            "hangout": "Hangout",
            "facebook": "Facebook",
            "email": "Email",
            "gmail": "Gmail",
            "ftp": "FTP",
            "sftp": "SFTP",
            "scp": "SCP",
            "netflix": "Netflix",
            "spotify": "Spotify",
            "vimeo": "Vimeo",
            "youtube": "YouTube",
            "aim": "AIM Chat",
            "voipbuster": "VOIPBuster",
            "bittorrent": "BitTorrent"
        }

        if app in app_map:
            return app_map[app]

        return "Unknown"

    # ===== 7. 6分类 =====
    if ISCX_TASK_MODE == 'category':

        if app in ["facebook", "skype", "hangout", "aim", "icq"]:
            if traffic_type == "audio":
                return "VoIP"
            elif traffic_type == "file":
                return "File Transfer"
            else:
                return "Chat"

        elif app in ["email","gmail"]:
            return "Email"

        elif app in ["youtube", "netflix", "spotify", "vimeo"]:
            return "Streaming"

        elif app in ["scp", "sftp", "ftp"]:
            return "File Transfer"

        elif app in ["bittorrent"]:
            return "P2P"

        elif app in ["voipbuster"]:
            return "VoIP"

        return "Unknown"


def get_clean_label(filename, dataset_name=DATASET_NAME):
    if dataset_name == "ustc":
        return get_ustc_label(filename)
    return get_iscx_label(filename)


def list_pcap_files(target_dir):
    return sorted([
        f for f in os.listdir(target_dir)
        if f.endswith('.pcap') or f.endswith('.pcapng')
    ])


def build_label_map(pcap_files, dataset_name=DATASET_NAME):
    label_to_id = {}

    for file_name in pcap_files:
        clean_name = get_clean_label(file_name, dataset_name)
        if clean_name == "Unknown":
            continue

        if clean_name not in label_to_id:
            label_to_id[clean_name] = len(label_to_id)

    return label_to_id


def extract_flow_features(pcap_path, label_id):
    try:
        streamer = nfstream.NFStreamer(
            source=pcap_path,
            statistical_analysis=True,
            splt_analysis=True,
            n_meters=0,
            performance_report=False,
            idle_timeout=60,
            active_timeout=300
        )
        df = streamer.to_pandas()
        if df.empty:
            return pd.DataFrame()

        feature_columns = select_feature_columns(df)
        if not feature_columns:
            return pd.DataFrame()

        df = df[feature_columns]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df = np.log1p(df)

        df['label'] = label_id
        return df
    except Exception as e:
        print(f"处理PCAP文件出错 {pcap_path}：{str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    dataset_config = get_dataset_config()
    target_dir = dataset_config["target_dir"]

    all_features = []

    if os.path.exists(target_dir):
        pcap_files = list_pcap_files(target_dir)
        label_to_id = build_label_map(pcap_files)

        if not pcap_files:
            print(f"目标目录下未找到 PCAP/PCAPNG 文件: {target_dir}")
            exit()

        if not label_to_id:
            print(f"未识别到有效类别，请检查 DATASET_NAME 和文件名: {target_dir}")
            exit()

        print(f"当前数据集: {dataset_config['description']}")
        print(f"检测到聚合后的类别总数: {len(label_to_id)}")
        print("类别映射详情:", json.dumps(label_to_id, indent=4, ensure_ascii=False))

        # 第二次遍历：提取特征
        for file_name in pcap_files:
            clean_name = get_clean_label(file_name)
            if clean_name == "Unknown":
                print(f"跳过未知类别文件: {file_name}")
                continue

            class_id = label_to_id[clean_name]
            pcap_path = os.path.join(target_dir, file_name)
            print(f"处理中: {file_name} -> 聚合标签: {clean_name} (ID: {class_id})")
            df = extract_flow_features(pcap_path, class_id)
            if not df.empty:
                all_features.append(df)

        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            final_df = normalize_feature_table(final_df)

            os.makedirs("./dataset", exist_ok=True)
            final_df.to_csv("./dataset/traffic_features.csv", index=False)

            with open("./dataset/label_map.json", "w") as f:
                json.dump(label_to_id, f, indent=4)

            print(f"特征提取成功！样本数: {len(final_df)}，聚合后类别数: {len(label_to_id)}")
        else:
            print("没有成功提取到任何特征。")
    else:
        print(f"路径 {target_dir} 不存在，请检查文件夹路径。")
