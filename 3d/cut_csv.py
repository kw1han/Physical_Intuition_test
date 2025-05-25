# import os
# import pandas as pd

# # 1. 获取所有图片主名（包含 _img 后缀）
# image_root = "/home/student0/Physical_Intuition_test/3d/Physion/Physion_image"
# image_names = set()

# for scene in os.listdir(image_root):
#     scene_dir = os.path.join(image_root, scene, "mp4s-redyellow")
#     if not os.path.isdir(scene_dir):
#         continue
#     for fname in os.listdir(scene_dir):
#         if fname.endswith(".png"):
#             main_name = os.path.splitext(fname)[0]
#             image_names.add(main_name)

# print(f"✅ 共提取图片主名数量：{len(image_names)}")

# # 2. 读取 CSV
# csv_path = os.path.join(image_root, "true_results_filtered.csv")
# df = pd.read_csv(csv_path)

# # 3. 构造 Stimulus Name → Actual Outcome 映射（保留唯一一行）
# stim_map = {}
# for _, row in df.iterrows():
#     stim_name = str(row["Stimulus Name"]).strip()
#     outcome = bool(int(row["Actual Outcome"]))
#     if stim_name not in stim_map:
#         stim_map[stim_name] = outcome  # 只保留第一次出现的记录

# # 4. 反向匹配：从图片名出发查 stim_name 或去掉 _img 后匹配
# matched_rows = []

# for img_name in image_names:
#     if img_name.endswith("_img"):
#         stim_name_core = img_name[:-4]
#     else:
#         stim_name_core = img_name

#     if stim_name_core in stim_map:
#         matched_rows.append({"Stimulus Name": img_name, "Actual Outcome": stim_map[stim_name_core]})

# # 5. 创建 DataFrame 并去重（如果之前重复还是保守保险）
# filtered_df = pd.DataFrame(matched_rows).drop_duplicates()

# # 6. 保存结果
# output_path = os.path.join(image_root, "true_results_filtered_matched.csv")
# filtered_df.to_csv(output_path, index=False)

# print(f"✅ 最终匹配成功 {len(filtered_df)} 条记录，已保存到: {output_path}")
import pandas as pd

# 读取已匹配的 CSV
csv_path = "/home/student0/Physical_Intuition_test/3d/Physion/Physion_image/true_results_filtered_matched.csv"
df = pd.read_csv(csv_path)

# 按 Stimulus Name 排序
df_sorted = df.sort_values(by="Stimulus Name")

# 保存排序后的结果
df_sorted.to_csv(csv_path, index=False)
print(f"✅ 已按 Stimulus Name 排序，并覆盖保存到: {csv_path}")
