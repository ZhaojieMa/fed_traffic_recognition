import os
import json


def export_filenames_to_json(folder_path, output_json_path="filenames.json", include_subfolders=False):
    """
    将指定文件夹下的所有文件名导出为JSON文件
    :param folder_path: 目标文件夹路径
    :param output_json_path: 输出的JSON文件路径（默认：当前目录下filenames.json）
    :param include_subfolders: 是否包含子文件夹中的文件（默认：不包含）
    """
    filenames_list = []

    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return

    if include_subfolders:
        # 递归遍历所有子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filenames_list.append(file)
    else:
        filenames_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 将列表写入JSON文件（ensure_ascii=False支持中文文件名，indent=4格式化输出）
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(filenames_list, json_file, ensure_ascii=False, indent=4)

    print(f"成功导出！共找到 {len(filenames_list)} 个文件")
    print(f"JSON文件路径：{os.path.abspath(output_json_path)}")

if __name__ == "__main__":
    TARGET_FOLDER = r"D:/USTC-TFC2016"

    # 2. 执行导出（include_subfolders=True 则包含子文件夹）
    export_filenames_to_json(
        folder_path=TARGET_FOLDER,
        output_json_path="filenames.json",
        include_subfolders=False
    )