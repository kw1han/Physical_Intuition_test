import os
import subprocess

# 配置参数
input_dir = "/home/student0/Physical_Intuition_test/3d/Physion/Physion/Support/mp4s-redyellow"    # 视频输入目录
output_dir = "/home/student0/Physical_Intuition_test/3d/Phytion_new/Support/mp4s-redyellow-trimmed"        # 输出目录
duration = "1.5"                      # 截取时长（秒）

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def batch_trim_videos():
    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4") and "pilot_towers" in filename:
            input_path = os.path.join(input_dir, filename)
            
            # 生成输出文件名（添加 _trimmed 后缀）
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_trimmed.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # FFmpeg 命令
            cmd = [
                "ffmpeg",
                "-y",                    # 覆盖已存在文件
                "-ss", "00:00:00",       # 起始时间
                "-i", input_path,        # 输入文件
                "-t", duration,          # 截取时长
                "-c", "copy",            # 使用流拷贝（无损快速）
                "-avoid_negative_ts", "make_zero",
                output_path
            ]
            
            try:
                # 执行命令
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
                print(f"成功处理: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"处理失败 {filename}: {e.stderr.decode()}")

if __name__ == "__main__":
    batch_trim_videos()
    print("处理完成！")