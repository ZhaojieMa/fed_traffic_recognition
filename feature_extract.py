import nfstream
import pandas as pd
import numpy as np
import os
import json
import warnings
import re

warnings.filterwarnings('ignore')

# 【新增】明确指定保留的特征列，剔除在 Non-VPN 中容易产生歧义或泄露的特征
# 例如：TTL 在不同操作系统下差异大，但在区分应用时无用；
# requested_server_name 在 VPN 中为空，在 Non-VPN 中有效，混合训练时会导致维度不匹配或误导
TARGET_FEATURES = [
    'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms',
    'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
    'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
    'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
    'bidirectional_packets', 'src2dst_packets', 'dst2src_packets',
    'bidirectional_bytes', 'src2dst_bytes', 'dst2src_bytes',
    # 核心统计特征：方差、最小值、最大值比平均值更能区分应用
    'bidirectional_duration_stddev', 'bidirectional_packet_size_mean',
    'bidirectional_packet_size_stddev', 'bidirectional_packet_size_min',
    'bidirectional_packet_size_max', 'bidirectional_packet_size_mode',
    'src2dst_packet_size_mean', 'src2dst_packet_size_stddev',
    'dst2src_packet_size_mean', 'dst2src_packet_size_stddev',
    'bidirectional_packet_time_mean', 'bidirectional_packet_time_stddev',
    'src2dst_packet_time_mean', 'src2dst_packet_time_stddev',
    'dst2src_packet_time_mean', 'dst2src_packet_time_stddev',
    # 流行为特征
    'src2dst_min_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_max_ps',
    'src2dst_min_ipi', 'src2dst_max_ipi', 'dst2src_min_ipi', 'dst2src_max_ipi',
    'flow_direction_changes', 'active_time', 'idle_time', 'active_packets', 'idle_packets'
]


def get_clean_label(filename):
    """
    针对 Non-VPN 数据集的标签清洗逻辑优化
    Non-VPN 数据集通常包含: Chat, Email, FTP, P2P, Streaming, VoIP, VPN_Chat...
    """
    name = file_name.lower()

    # 1. 去除 VPN 前缀
    if name.startswith("vpn_"):
        name = name.replace("vpn_", "", 1)

    # 2. 特殊处理：有些文件名没有下划线分隔，如 facebookchat1
    # 我们先把已知的关键词标准化
    keywords = ["aim_chat", "aimchat", "email", "facebook_chat", "facebookchat",
                "facebook_audio", "facebook_video", "bittorrent", "ftps",
                "hangouts_audio", "hangouts_chat"]

    # 统一转换常见的连写
    name = name.replace("aimchat", "aim_chat")
    name = name.replace("facebookchat", "facebook_chat")

    # 3. 提取核心关键词 (匹配到即停止)
    for kw in ["aim_chat", "email", "facebook_chat", "facebook_audio",
               "facebook_video", "bittorrent", "ftps", "hangouts_audio", "hangouts_chat"]:
        if kw in name:
            return kw

    return "others"


def extract_flow_features(pcap_path, label_id):
    try:
        streamer = nfstream.NFStreamer(
            source=pcap_path,
            statistical_analysis=True,
            splt_analysis=True,  # 开启包长/时间间隔序列分析，这对 Non-VPN 至关重要
            n_meters=0,
            performance_report=False,
            idle_timeout=60,
            active_timeout=300
        )
        df = streamer.to_pandas()
        if df.empty:
            return pd.DataFrame()

        # --- 特征工程核心修正 ---

        # 1. 移除所有标识符和非数值列
        drop_cols = [
            'id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
            'dst_ip', 'dst_mac', 'dst_oui', 'dst_port',
            'application_name', 'application_category_name', 'category_name',
            'client_fingerprint', 'server_fingerprint', 'requested_server_name',  # 移除 SNI，防止 Non-VPN 作弊
            'application_is_guessed', 'vlan_id', 'tunnel_id',
            'entry_type'  # 移除 nfstream 内部字段
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # 2. 强制对齐特征列
        # 确保只保留 TARGET_FEATURES 中定义的列，防止不同数据集特征维度不一致
        missing_cols = [col for col in TARGET_FEATURES if col not in df.columns]
        if missing_cols:
            # 如果某些特征不存在（版本差异），创建空列
            for col in missing_cols:
                df[col] = 0.0

        df = df[TARGET_FEATURES]  # 只保留目标特征

        # 3. 仅保留数值列
        df = df.select_dtypes(include=['number'])

        # 4. 处理无穷大和空值
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 5. 【关键】对流量特征进行 log(1+x) 处理
        # Non-VPN 流量中，文件大小差异极大（Email vs FTP），Log 处理能极大提升区分度
        df = np.log1p(df)

        df['label'] = label_id
        return df
    except Exception as e:
        print(f"处理PCAP文件出错 {pcap_path}：{str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    # 请根据实际路径切换
    target_dir = "D:/VPN-NonVPN-PCAPs-01"

    all_features = []
    label_to_id = {}
    next_id = 0

    if os.path.exists(target_dir):
        pcap_files = [f for f in os.listdir(target_dir) if f.endswith('.pcap') or f.endswith('.pcapng')]

        # 第一次遍历：确定类别映射
        for file_name in pcap_files:
            clean_name = get_clean_label(file_name)
            if clean_name not in label_to_id:
                label_to_id[clean_name] = next_id
                next_id += 1

        print(f"检测到聚合后的类别总数: {next_id}")
        print("类别映射详情:", label_to_id)

        # 第二次遍历：提取特征
        for file_name in pcap_files:
            clean_name = get_clean_label(file_name)
            class_id = label_to_id[clean_name]
            pcap_path = os.path.join(target_dir, file_name)
            print(f"处理中: {file_name} -> 聚合标签: {clean_name} (ID: {class_id})")
            df = extract_flow_features(pcap_path, class_id)
            if not df.empty:
                all_features.append(df)

        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)

            os.makedirs("./dataset", exist_ok=True)
            final_df.to_csv("./dataset/traffic_features.csv", index=False)

            # 保存元数据
            with open("./dataset/label_map.json", "w") as f:
                json.dump(label_to_id, f, indent=4)

            print(f"特征提取成功！样本数: {len(final_df)}，聚合后类别数: {next_id}")
            print("提示：Non-VPN 数据由于加密协议（HTTPS）同质化严重，统计特征区分度天然低于 VPN。")