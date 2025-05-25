import os
import cv2

# 原始视频根目录
src_root = "/home/student0/Physical_Intuition_test/3d/Physion/Physion"
# 目标图片根目录
dst_root = "/home/student0/Physical_Intuition_test/3d/Physion/Physion_image"

for scene in os.listdir(src_root):
    scene_dir = os.path.join(src_root, scene, "mp4s-redyellow")
    if not os.path.isdir(scene_dir):
        continue

    # 目标场景目录
    dst_scene_dir = os.path.join(dst_root, scene, "mp4s-redyellow")
    os.makedirs(dst_scene_dir, exist_ok=True)

    for filename in os.listdir(scene_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(scene_dir, filename)
        img_name = filename.replace(".mp4", ".png")
        img_path = os.path.join(dst_scene_dir, img_name)

        # 跳过已存在图片
        if os.path.exists(img_path):
            print(f"已存在，跳过: {img_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(img_path, frame)
            print(f"保存起始帧: {img_path}")
        else:
            print(f"读取失败: {video_path}")