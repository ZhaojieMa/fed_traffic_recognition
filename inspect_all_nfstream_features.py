import nfstream
import os
import json
import warnings
import re
from collections import defaultdict

warnings.filterwarnings('ignore')

# 数据集切换：
#   "iscx" -> D:/VPN-NonVPN-PCAPs-01
#   "ustc" -> D:/USTC-TFC2016
DATASET_NAME = "ustc"
ISCX_TASK_MODE = 'application'  # Encapsulation|category|application

DATASET_CONFIGS = {
    "iscx": {
        "target_dir": "D:/VPN-NonVPN-PCAPs-01",
        "description": "ISCX VPN-NonVPN",
        "output_json": "./config/all_nfstream_features.json"
    },
    "ustc": {
        "target_dir": "D:/USTC-TFC2016",
        "description": "USTC-TFC2016 10-class",
        "output_json": "./config/ustc_nfstream_features.json"
    }
}

# 只用于查看 NFStream 能提取出的全部字段
OUTPUT_JSON = DATASET_CONFIGS[DATASET_NAME]["output_json"]
TARGET_DIR = DATASET_CONFIGS[DATASET_NAME]["target_dir"]

SCAN_LIMIT = None

DROP_COLS = [
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

        elif app in ["email", "gmail"]:
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


def build_label_map(pcap_files, dataset_name=DATASET_NAME):
    label_to_id = {}
    next_id = 0

    for file_name in pcap_files:
        clean_name = get_clean_label(file_name, dataset_name)
        if clean_name == "Unknown":
            continue

        if clean_name not in label_to_id:
            label_to_id[clean_name] = next_id
            next_id += 1

    return label_to_id


def infer_column_types(df):
    """
    返回当前 df 中每个字段的类型信息。
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    return numeric_cols, non_numeric_cols


def scan_one_pcap(pcap_path):
    """
    解析单个 PCAP，返回：
    1. 所有字段名
    2. 数值字段名
    3. 非数值字段名
    4. 每个字段的数据类型
    """
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
        return {
            "empty": True,
            "columns": [],
            "numeric_columns": [],
            "non_numeric_columns": [],
            "dtypes": {}
        }

    numeric_cols, non_numeric_cols = infer_column_types(df)

    return {
        "empty": False,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def main():
    dataset_config = get_dataset_config()
    target_dir = dataset_config["target_dir"]
    output_json = dataset_config["output_json"]

    if not os.path.exists(target_dir):
        print(f"目标目录不存在: {target_dir}")
        return

    pcap_files = sorted([
        f for f in os.listdir(target_dir)
        if f.endswith('.pcap') or f.endswith('.pcapng')
    ])

    if SCAN_LIMIT is not None:
        pcap_files = pcap_files[:SCAN_LIMIT]

    if not pcap_files:
        print(f"目标目录下未找到 PCAP/PCAPNG 文件: {target_dir}")
        return

    label_to_id = build_label_map(pcap_files)

    all_columns = set()
    numeric_columns = set()
    non_numeric_columns = set()

    column_dtypes = defaultdict(set)
    column_appear_count = defaultdict(int)
    column_files = defaultdict(list)

    per_file_summary = []
    empty_files = []
    error_files = []

    print(f"当前数据集: {dataset_config['description']}")
    print(f"开始扫描 NFStream 可提取字段，共 {len(pcap_files)} 个文件...")

    for idx, file_name in enumerate(pcap_files, start=1):
        clean_name = get_clean_label(file_name)
        pcap_path = os.path.join(target_dir, file_name)

        print(f"[{idx}/{len(pcap_files)}] 扫描: {file_name} -> {clean_name}")

        try:
            result = scan_one_pcap(pcap_path)

            if result["empty"]:
                empty_files.append(file_name)
                per_file_summary.append({
                    "file": file_name,
                    "label": clean_name,
                    "empty": True,
                    "num_columns": 0,
                    "num_numeric_columns": 0,
                    "num_non_numeric_columns": 0
                })
                continue

            cols = result["columns"]
            nums = result["numeric_columns"]
            non_nums = result["non_numeric_columns"]

            all_columns.update(cols)
            numeric_columns.update(nums)
            non_numeric_columns.update(non_nums)

            for col in cols:
                column_appear_count[col] += 1
                column_files[col].append(file_name)

            for col, dtype in result["dtypes"].items():
                column_dtypes[col].add(dtype)

            per_file_summary.append({
                "file": file_name,
                "label": clean_name,
                "empty": False,
                "num_columns": len(cols),
                "num_numeric_columns": len(nums),
                "num_non_numeric_columns": len(non_nums),
                "columns": cols
            })

        except Exception as e:
            print(f"扫描出错: {file_name} -> {str(e)}")
            error_files.append({
                "file": file_name,
                "error": str(e)
            })

    all_columns = sorted(all_columns)
    numeric_columns = sorted(numeric_columns)
    non_numeric_columns = sorted(non_numeric_columns)

    dropped_columns = sorted([c for c in all_columns if c in DROP_COLS])
    retained_candidate_columns = sorted([c for c in all_columns if c not in DROP_COLS])
    retained_numeric_candidate_columns = sorted([
        c for c in numeric_columns
        if c not in DROP_COLS
    ])

    retained_non_numeric_candidate_columns = sorted([
        c for c in non_numeric_columns
        if c not in DROP_COLS
    ])

    output = {
        "description": (
            "This file lists all columns that NFStream can extract from the scanned PCAP files. "
            "No TARGET_FEATURES filtering is applied here. "
            "It is intended for inspecting the complete feature space before manually selecting or generating a final schema."
        ),
        "dataset_name": DATASET_NAME,
        "dataset_description": dataset_config["description"],
        "task_mode": ISCX_TASK_MODE if DATASET_NAME == "iscx" else "10_class",
        "iscx_task_mode": ISCX_TASK_MODE,
        "target_dir": target_dir,
        "num_scanned_files": len(pcap_files),
        "num_empty_files": len(empty_files),
        "num_error_files": len(error_files),
        "num_all_columns": len(all_columns),
        "num_numeric_columns": len(numeric_columns),
        "num_non_numeric_columns": len(non_numeric_columns),

        "all_columns": all_columns,
        "numeric_columns": numeric_columns,
        "non_numeric_columns": non_numeric_columns,

        "drop_cols": DROP_COLS,
        "dropped_columns_found": dropped_columns,
        "retained_candidate_columns": retained_candidate_columns,
        "retained_numeric_candidate_columns": retained_numeric_candidate_columns,
        "retained_non_numeric_candidate_columns": retained_non_numeric_candidate_columns,

        # 兼容旧字段名。
        "original_drop_cols": DROP_COLS,
        "original_dropped_columns_found": dropped_columns,
        "original_retained_candidate_columns": retained_candidate_columns,
        "original_retained_numeric_candidate_columns": retained_numeric_candidate_columns,
        "original_retained_non_numeric_candidate_columns": retained_non_numeric_candidate_columns,

        "column_dtypes": {
            col: sorted(list(dtypes))
            for col, dtypes in sorted(column_dtypes.items())
        },
        "column_appear_count": {
            col: int(column_appear_count[col])
            for col in all_columns
        },
        "label_to_id": label_to_id,
        "empty_files": empty_files,
        "error_files": error_files,
        "per_file_summary": per_file_summary
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\n扫描完成")
    print(f"输出 JSON: {output_json}")
    print(f"所有字段数: {len(all_columns)}")
    print(f"数值字段数: {len(numeric_columns)}")
    print(f"非数值字段数: {len(non_numeric_columns)}")
    print(f"按当前剔除规则会删除的字段数: {len(dropped_columns)}")
    print(f"按当前剔除规则保留的数值候选字段数: {len(retained_numeric_candidate_columns)}")


if __name__ == "__main__":
    main()
