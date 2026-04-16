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
    pcap_files = {
        "D:/VPN-PCAPS-01/vpn_aim_chat1a.pcap": "VPN_Chat1a",
        "D:/VPN-PCAPS-01/vpn_aim_chat1b.pcap": "VPN_Chat1b",
        "D:/VPN-PCAPs-01/vpn_bittorrent.pcap": "VPN_Bittorrent",
        "D:/VPN-PCAPs-01/vpn_email2a.pcap": "VPN_Email2a",
        "D:/VPN-PCAPs-01/vpn_email2b.pcap": "VPN_Email2b",
        "D:/VPN-PCAPs-01/vpn_facebook_audio2.pcap": "VPN_Facebook_Audio2a",
        "D:/VPN-PCAPs-01/vpn_facebook_chat1a.pcap": "VPN_Facebook_Chat1a",
        "D:/VPN-PCAPs-01/vpn_facebook_chat1b.pcap": "VPN_Facebook_Chat1b",
        "D:/VPN-PCAPs-01/vpn_ftps_A.pcap": "VPN_Ftps_A",
        "D:/VPN-PCAPs-01/vpn_ftps_B.pcap": "VPN_Ftps_B",
        "D:/VPN-PCAPs-01/vpn_hangouts_audio1.pcap": "VPN_Hangouts_Audio1",
        "D:/VPN-PCAPs-01/vpn_hangouts_audio2.pcap": "VPN_Hangouts_Audio2",
        "D:/VPN-PCAPs-01/vpn_hangouts_chat1a.pcap": "VPN_Hangouts_Chat1a",
        "D:/VPN-PCAPs-01/vpn_hangouts_chat1b.pcap": "VPN_Hangouts_Chat1b",
    }

    all_features = []
    for pcap_path, label in pcap_files.items():
        if os.path.exists(pcap_path):
            df = extract_flow_features(pcap_path, label)
            if not df.empty:
                all_features.append(df)
        else:
            print(f"警告: 找不到文件 {pcap_path}")

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
