import os
import json

def extract_and_integrate_results(video_dir: str, metadata_dir: str, output_file: str):
    # 存储结果的字典
    results = {}
    modified_count = 0  # 计数器，跟踪成功修改的文件名数量

    # 遍历视频目录，获取所有 mp4 文件
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                # 分别去掉 -redyellow 和 _img_trimmed.mp4
                modified_name = file.replace('-redyellow', '').replace('_img_trimmed.mp4', '.mp4')
                # 存储修改后的文件名
                results[modified_name] = None  # 初始化为 None，稍后填充
                modified_count += 1  # 增加计数器

    # 只输出成功修改的文件名数量
    print(f"成功修改的文件名数量: {modified_count}")

    # 遍历 metadata.json 文件
    for root, dirs, files in os.walk(metadata_dir):
        for file in files:
            if file == 'metadata.json':
                with open(os.path.join(root, file), 'r') as f:
                    metadata = json.load(f)
                    # 假设 metadata 是一个列表
                    for entry in metadata:
                        stimulus_name = entry.get("stimulus_name").strip()  # 去除多余空格
                        if stimulus_name in results:
                            results[stimulus_name] = entry.get("does_target_contact_zone")

    # 只输出最终结果
    print("最终结果:")
    final_results = {name: contact_zone for name, contact_zone in results.items() if contact_zone is not None}

    # 保存结果到 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")

# 调用函数
video_path = '/home/student0/Physical_Intuition_test/3d/Phytion_new/Collide/mp4s-redyellow-trimmed'
metadata_path = '/home/student0/Physical_Intuition_test/3D_physion/physics-benchmarking-neurips2021/stimuli/generation/configs/collide'
output_path = '/home/student0/Physical_Intuition_test/3d/collide_results.json'

extract_and_integrate_results(video_path, metadata_path, output_path)