import nfstream
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

def extract_flow_features(pcap_path, label):
    try:
        streamer = nfstream.NFStreamer(
            source=pcap_path, statistical_analysis=True, splt_analysis=True,
            n_meters=0, performance_report=False, idle_timeout=60, active_timeout=300
        )
        df = streamer.to_pandas()
        if df.empty: return pd.DataFrame()

        # 严格移除标识符特征，防止模型通过IP或端口作弊（Data Leakage）
        drop_cols = [
            'id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
            'dst_ip', 'dst_mac', 'dst_oui', 'dst_port',
            'application_name', 'application_category_name',
            'category_name', 'client_fingerprint', 'server_fingerprint',
            'requested_server_name', 'application_is_guessed', 'vlan_id'
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # 仅保留严格的纯数值型统计特征
        df = df.select_dtypes(include=['number'])
        df['label'] = label
        return df
    except Exception as e:
        print(f"处理PCAP文件出错 {pcap_path}：{str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    pcap_root_dir = "D:/VPN-PCAPs-01"

    all_features = []

    # 2. 自动遍历目录下所有文件
    if os.path.exists(pcap_root_dir):
        # 获取目录下所有以 .pcap 结尾的文件
        pcap_files = [f for f in os.listdir(pcap_root_dir) if f.endswith('.pcap')]

        print(f"在目录 {pcap_root_dir} 中找到 {len(pcap_files)} 个 PCAP 文件。")

        for file_name in pcap_files:
            # 构建完整的文件路径
            pcap_path = os.path.join(pcap_root_dir, file_name)

            # 自动生成标签：例如 "vpn_aim_chat1a.pcap" -> "vpn_aim_chat1a"
            # 你也可以根据需要使用 split('_') 等进一步简化标签
            label = os.path.splitext(file_name)[0]

            print(f"正在处理: {file_name} (标签: {label})")

            df = extract_flow_features(pcap_path, label)
            if not df.empty:
                all_features.append(df)
    else:
        print(f"错误: 找不到目录 {pcap_root_dir}")

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df = final_df.replace([float('inf'), float('-inf')], float('nan'))
        # 修复：直接填0，不要在划分前使用全局均值填充，避免数据泄露
        final_df = final_df.fillna(0.0)

        os.makedirs("./dataset", exist_ok=True)
        final_df.to_csv("./dataset/traffic_features.csv", index=False, encoding='utf-8')
        print(f"特征提取完成！样本数：{len(final_df)}，特征维度：{final_df.shape[1] - 1}")
    else:
        print("未提取到任何特征，请检查PCAP路径。")
