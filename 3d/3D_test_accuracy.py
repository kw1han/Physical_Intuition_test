import os
import json
import random
from pathlib import Path
import argparse
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import re
import cv2
from datetime import datetime

class Logger:
    """日志记录类，同时输出到终端和文件"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"evaluation_log_{timestamp}.txt"
        
        # 创建日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def log(self, message: str):
        """记录消息到终端和文件"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_section(self, title: str):
        """记录带分隔符的章节标题"""
        separator = "=" * 50
        message = f"\n{separator}\n{title}\n{separator}\n"
        self.log(message)
    
    def log_subsection(self, title: str):
        """记录带分隔符的子章节标题"""
        separator = "-" * 30
        message = f"\n{separator}\n{title}\n{separator}\n"
        self.log(message)

def encode_image_to_base64(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class PhysicalIntuition3DEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "deepseek-vl2",
                 api_key: Optional[str] = None,
                 base_url: str = "https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions",
                 output_dir: str = "/home/student0/Physical_Intuition_test/3d/result_3d",
                 game_type: Optional[str] = None):
        """
        初始化三维物理直觉评估器
        
        Args:
            data_root: 数据根目录
            model_name: 要测试的模型名称
            api_key: API密钥
            base_url: API基础URL
            output_dir: 结果输出目录
            game_type: 要测试的游戏类型，如果为None则测试所有类型
        """
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.game_type = game_type
        
        # 初始化日志记录器
        self.logger = Logger(self.output_dir)
        
        # 初始化API客户端
        self.client = OpenAI(
            api_key="sk-uruhfkcrzkebvehlyoeradzzwentjyjpqbteiryddkkpahpe",
            base_url=base_url,
        )
        
        # 初始化结果记录
        self.results = []
        self.response_times = []
        
        # 在初始化时就获取所有视频文件并缓存
        self.logger.log("正在初始化视频文件缓存...")
        self.all_videos = self.find_video_files()
        self.logger.log(f"已缓存 {len(self.all_videos)} 个视频文件")
        
        # 按游戏类型分组缓存视频（不再区分成功/失败）
        self.videos_by_type = {}
        for video in self.all_videos:
            game_type = video.parent.parent.name
            
            if game_type not in self.videos_by_type:
                self.videos_by_type[game_type] = []
            
            self.videos_by_type[game_type].append(video)
        
        # 游戏类型映射到描述
        self.game_type_desc = {
            "Collide": "collision scenario",
            "Contain": "containment scenario",
            "Dominoes": "domino scenario",
            "Drape": "draping scenario",
            "Drop": "dropping scenario",
            "Link": "linking scenario",
            "Roll": "rolling scenario",
            "Support": "support scenario"
        }
        
        # 加载场景特定提示
        self.scenario_prompts = {}
        self._load_scenario_prompts()

    def _load_scenario_prompts(self):
        """加载场景特定提示"""
        prompt_dir = Path(self.data_root).parent / "prompt"
        if not prompt_dir.exists():
            self.logger.log(f"警告: 提示目录不存在: {prompt_dir}")
            return
            
        # 遍历所有游戏类型
        for game_type in self.game_type_desc.keys():
            # 尝试加载对应的提示文件（小写文件名）
            prompt_file = prompt_dir / f"{game_type.lower()}.txt"
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.scenario_prompts[game_type] = f.read().strip()
                        self.logger.log(f"成功加载 {game_type} 的提示文件")
                except Exception as e:
                    self.logger.log(f"警告: 无法加载 {game_type} 的提示文件: {e}")
            else:
                self.logger.log(f"警告: 未找到 {game_type} 的提示文件: {prompt_file}")
                # 使用默认提示
                self.scenario_prompts[game_type] = f"""
You will see several video frames from a {self.game_type_desc[game_type]}. Please analyze the physical interaction between objects and predict the outcome.

Task Requirements:
1. Analyze the motion and interaction of objects in the scene
2. Consider physical principles such as gravity, momentum, and collision
3. Make a prediction about whether the intended interaction will be successful
4. Start your answer with YES or NO, followed by your reasoning

Please provide a clear judgment and explain your physical reasoning process.
"""

    def find_video_files(self, success_only: bool = None, game_types: List[str] = None) -> List[Path]:
        """
        查找符合条件的视频文件
        
        Args:
            success_only: 不再使用此参数
            game_types: 要筛选的游戏类型列表
            
        Returns:
            符合条件的视频文件路径列表
        """
        all_videos = []
        
        if not self.data_root.exists():
            self.logger.log("错误: 数据目录不存在")
            return all_videos
        
        self.logger.log(f"正在搜索数据目录: {self.data_root}")
        
        for game_folder in self.data_root.iterdir():
            if not game_folder.is_dir():
                self.logger.log(f"跳过非目录文件: {game_folder}")
                continue
                
            game_type = game_folder.name
            self.logger.log(f"正在处理游戏类型目录: {game_type}")
            
            if game_types and game_type not in game_types:
                self.logger.log(f"跳过非指定游戏类型: {game_type}")
                continue
                
            trimmed_dir = game_folder / "mp4s-redyellow-trimmed"
            if not trimmed_dir.exists():
                self.logger.log(f"警告: 在 {game_type} 目录中未找到 mp4s-redyellow-trimmed 子目录")
                continue
                
            video_count = 0
            for video_file in trimmed_dir.glob("*.mp4"):
                # 移除成功状态的判断，直接添加所有视频
                self.logger.log(f"找到视频文件: {video_file.name}")
                
                all_videos.append(video_file)
                video_count += 1
                
            self.logger.log(f"在 {game_type} 目录中找到 {video_count} 个视频文件")
        
        self.logger.log(f"总共找到 {len(all_videos)} 个视频文件")
        return all_videos

    def find_image_files(self) -> List[Path]:
        """
        查找符合条件的图像文件
        
        Returns:
            符合条件的图像文件路径列表
        """
        all_images = []
        
        if not self.data_root.exists():
            self.logger.log("错误: 数据目录不存在")
            return all_images
        
        self.logger.log(f"正在搜索数据目录: {self.data_root}")
        
        # 假设physion_image的目录结构与原来类似，但是存储的是图像而不是视频
        for game_folder in self.data_root.iterdir():
            if not game_folder.is_dir():
                self.logger.log(f"跳过非目录文件: {game_folder}")
                continue
            
            game_type = game_folder.name
            self.logger.log(f"正在处理游戏类型目录: {game_type}")
            
            if game_types and game_type not in game_types:
                self.logger.log(f"跳过非指定游戏类型: {game_type}")
                continue
            
            # 修改为直接查找图像文件
            for image_file in game_folder.glob("*.png"):  # 或者其他图像格式 (*.jpg, *.jpeg)
                self.logger.log(f"找到图像文件: {image_file.name}")
                all_images.append(image_file)
            
        self.logger.log(f"总共找到 {len(all_images)} 个图像文件")
        return all_images

    def extract_frames_from_video(self, video_path: Path, frame_count: int = 4) -> List[str]:
        """
        从视频中提取指定数量的帧并保存为临时文件
        
        Args:
            video_path: 视频文件路径
            frame_count: 要提取的帧数
            
        Returns:
            提取的帧文件路径列表
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(total_frames * i / frame_count) for i in range(frame_count)]
        
        extracted_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_path = f"/tmp/frame_{video_path.stem}_{idx}.png"
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
        
        cap.release()
        return extracted_frames
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像文件编码为Base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Base64编码的图像字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def prepare_shot_examples(self, shot_count: int) -> List[Dict]:
        """准备示例视频"""
        examples = []
        
        if not self.all_videos:
            self.logger.log("警告: 没有找到任何视频文件")
            return examples
        
        # 动态选择不同场景类型的示例
        selected_types = set()
        while len(examples) < shot_count:
            # 优先选择未出现的场景类型
            available_types = [t for t in self.videos_by_type.keys() if t not in selected_types]
            if not available_types:
                available_types = list(self.videos_by_type.keys())
            
            # 随机选择一个类型
            game_type = random.choice(available_types)
            videos = self.videos_by_type[game_type]
            
            if not videos:
                continue
            
            # 随机选择一个视频
            video = random.choice(videos)
            
            # 为示例设置随机的成功状态（或根据实际情况设置）
            is_success = random.choice([True, False])
            
            examples.append({
                "video_path": str(video),
                "is_success": is_success,
                "game_type": game_type,
                "description": f"在这个{self.game_type_desc.get(game_type, game_type)}中，"
                              f"红色物体{'成功' if is_success else '未能'}完成预期运动。"
            })
            selected_types.add(game_type)
        
        return examples[:shot_count]

    def build_prompt(self, test_video: Path, shot_count: int) -> Tuple[List, bool, str]:
        """
        构建提示
        
        Args:
            test_video: 要测试的视频
            shot_count: 示例数量
            
        Returns:
            提示消息列表，真实结果和游戏类型
        """
        # 从视频中提取帧
        extracted_frames = self.extract_frames_from_video(test_video)
        base64_images = [self.encode_image_to_base64(frame) for frame in extracted_frames]
        
        # 获取游戏类型
        game_type = test_video.parent.parent.name
        
        # 如果您有特定的真实结果判断逻辑，可以在这里添加
        # 例如，可以通过其他方式确定真实结果，而不是依赖文件名
        true_result = True  # 或者设置为您需要的默认值
        
        messages = []
        
        # 系统提示 - 使用场景特定提示
        system_message = {
            "role": "system", 
            "content": self.scenario_prompts.get(game_type, "你是一个具有强大物理直觉的AI助手。你需要基于物理场景图像预测红色球是否会到达绿色目标区域。在回答时，请先明确回答YES或NO，然后再提供解释。")
        }
        messages.append(system_message)
        
        # 添加few-shot示例
        if shot_count > 0:
            examples = self.prepare_shot_examples(shot_count)
            for i, example in enumerate(examples):
                try:
                    example_frames = self.extract_frames_from_video(Path(example['video_path']))
                    example_images = [self.encode_image_to_base64(frame) for frame in example_frames]
            
                    example_content = [
                        {"type": "text", "text": f"示例{i+1}视频的关键帧："},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example_images[0]}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example_images[1]}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example_images[2]}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example_images[3]}"}},
                        {"type": "text", "text": example["description"]}
                    ]
                    messages.append({"role": "user", "content": example_content})
                    messages.append({"role": "assistant", "content": f"{'YES' if example['is_success'] else 'NO'}, 我理解。在这个{self.game_type_desc.get(example['game_type'], example['game_type'])}中，红色物体{'成功' if example['is_success'] else '未能'}完成预期运动。"})
                except Exception as e:
                    self.logger.log(f"警告: 处理示例视频时出错: {e}")
                    continue
        
        # 添加测试问题
        question_text = f"基于这个{self.game_type_desc.get(game_type, game_type)}的关键帧，红色物体最终会{'成功' if true_result else '未能'}完成预期运动吗？请先明确回答YES或NO，然后详细解释你的推理过程，包括影响物体运动的物理因素。"
        
        if shot_count > 0:
            question_text = "基于之前的示例和你的物理知识，" + question_text
        
        question_content = [
            {"type": "text", "text": question_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_images[0]}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_images[1]}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_images[2]}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_images[3]}"}}
        ]
        
        messages.append({"role": "user", "content": question_content})
        
        return messages, true_result, game_type
    
    def call_model(self, messages: List[Dict], temperature: float = 0.0) -> Dict:
        """
        调用模型
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            模型响应
        """
        try:
            self.logger.log("正在调用API...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            self.logger.log("API调用成功")
            return response
        except Exception as e:
            self.logger.log(f"API调用错误: {str(e)}")
            # 记录更多详细错误信息
            import traceback
            self.logger.log(traceback.format_exc())
            
            time.sleep(3)
            try:
                self.logger.log("正在重试API调用...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                self.logger.log("重试成功")
                return response
            except Exception as e2:
                self.logger.log(f"重试失败: {str(e2)}")
                self.logger.log(traceback.format_exc())
                return None
    
    def parse_prediction(self, response_text: str) -> bool:
        """
        从响应文本中解析预测结果
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            预测结果（True表示成功，False表示失败）
        """
        # 提取响应的前100个字符进行分析
        first_part = response_text[:100].upper()
        
        # 明确检查YES或NO在开头
        if first_part.startswith("YES"):
            return True
        elif first_part.startswith("NO"):
            return False
            
        # 检查是否包含明确的"YES"或"NO"标记
        response_upper = response_text.upper()
        
        # 匹配YES和NO周围更明确的模式
        yes_patterns = [
            r"\bYES\b", r"\bYES,", r"\bYES:", r"\bYES\.", 
            r"THE ANSWER IS YES", r"MY ANSWER IS YES"
        ]
        
        no_patterns = [
            r"\bNO\b", r"\bNO,", r"\bNO:", r"\bNO\.", 
            r"THE ANSWER IS NO", r"MY ANSWER IS NO"
        ]
        
        import re
        
        # 检查YES模式
        for pattern in yes_patterns:
            if re.search(pattern, response_upper):
                return True
            
        # 检查NO模式
        for pattern in no_patterns:
            if re.search(pattern, response_upper):
                return False
        
        # 更广泛的关键词匹配
        positive_indicators = [
            "WILL REACH", "CAN REACH", "SUCCESSFULLY", "WILL MAKE IT", 
            "ABLE TO REACH", "REACHES THE TARGET"
        ]
        
        negative_indicators = [
            "WILL NOT REACH", "WON'T REACH", "CANNOT REACH", "CAN'T REACH",
            "FAILS TO REACH", "WILL MISS", "UNABLE TO REACH", "DOESN'T REACH"
        ]
        
        # 先检查否定指示词
        for indicator in negative_indicators:
            if indicator in response_upper:
                return False
            
        # 然后检查肯定指示词
        for indicator in positive_indicators:
            if indicator in response_upper:
                return True
        
        # 如果以上都未匹配，检查YES/NO在文本中的位置
        yes_pos = response_upper.find("YES")
        no_pos = response_upper.find("NO")
        
        if yes_pos >= 0 and no_pos >= 0:
            return yes_pos < no_pos
        elif yes_pos >= 0:
            return True
        elif no_pos >= 0:
            return False
        
        # 默认返回False（保守预测）
        return False
    
    def evaluate_video(self, video_path: Path, shot_count: int) -> Dict:
        """评估单个视频"""
        self.logger.log(f"评估: {video_path.name}")
        
        messages, true_result, game_type = self.build_prompt(video_path, shot_count)
        if not messages:
            return None
        
        start_time = time.time()
        response = self.call_model(messages)
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        if not response:
            return None
        
        response_text = response.choices[0].message.content
        prediction = self.parse_prediction(response_text)
        
        result = {
            "video": video_path.name,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text,
            "response_time": response_time
        }
        
        self.results.append(result)
        return result
    
    def select_diverse_videos(self, num=20):
        """选择多样化的视频样本"""
        # 使用缓存的视频列表
        if not self.all_videos:
            self.logger.log("警告: 没有找到任何视频文件")
            return []
        
        # 打印所有可用的游戏类型
        available_types = set(v.parent.parent.name for v in self.all_videos)
        self.logger.log(f"可用的游戏类型: {available_types}")
        self.logger.log(f"当前设置的游戏类型: {self.game_type}")
        
        # 如果指定了游戏类型，只从该类型的视频中选择
        if self.game_type:
            candidates = [v for v in self.all_videos if v.parent.parent.name == self.game_type]
            self.logger.log(f"筛选出游戏类型 {self.game_type} 的视频: {len(candidates)} 个")
        else:
            candidates = self.all_videos.copy()
            self.logger.log(f"使用所有视频: {len(candidates)} 个")
            return candidates
        
        # 检查是否有足够的视频
        if len(candidates) < num:
            self.logger.log(f"警告: 可用的视频数量({len(candidates)})少于请求的数量({num})，将使用所有可用视频")
            return candidates
        
        # 选择多样化视频逻辑...（保留原有的多样化选择代码）
        
        # 返回选中的视频
        return candidates

    def run_evaluation(self, num_images=20, shot_counts=[0,1,2]):
        """运行评估"""
        # 选择图像
        selected_images = self.select_diverse_images(num_images)
        
        if not selected_images:
            self.logger.log("错误: 没有找到任何图像文件，请检查数据目录")
            return
        
        self.logger.log(f"找到 {len(selected_images)} 个图像文件")
        if self.game_type:
            self.logger.log(f"仅测试 {self.game_type} 类型的图像")
        
        # 按游戏类型统计选中的图像
        images_by_type = {}
        for image in selected_images:
            game_type = image.parent.name
            if game_type not in images_by_type:
                images_by_type[game_type] = []
            images_by_type[game_type].append(image)
        
        self.logger.log("\n选中图像的分布:")
        for game_type, images in images_by_type.items():
            self.logger.log(f"{game_type}: {len(images)} 个图像")
        
        # 对每个shot设置进行评估
        for shot in shot_counts:
            self.logger.log_section(f"当前Shot设置: {shot}")
            self.results = []  # 重置结果列表
            for image in selected_images:
                result = self.evaluate_image(image, shot)
                if result:
                    self.logger.log(f"结果: {result['correct']} ({image.parent.name})")
            
            self.save_results(f"physion_image_results_{shot}shot.json")
        
        if self.results:
            self.analyze_results()
        else:
            self.logger.log("警告: 没有生成任何结果，跳过分析")

    def analyze_results(self):
        """分析结果"""
        if not self.results:
            self.logger.log("警告: 没有结果可分析")
            return
        
        df = pd.DataFrame(self.results)
        
        # 检查必要的列是否存在
        required_columns = ['game_type', 'shot_count', 'correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.log(f"错误: 结果中缺少必要的列: {missing_columns}")
            return
        
        # 总体统计
        accuracy = df.groupby(['game_type', 'shot_count'])['correct'].mean().unstack()
        
        # 可视化
        plt.figure(figsize=(12,6))
        accuracy.plot(kind='bar')
        plt.title('三维物理直觉准确率分析')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / '3d_accuracy.png')
        
        # 保存详细结果
        df.to_csv(self.output_dir / '3d_results.csv', index=False)
        
        # 打印统计信息
        self.logger.log("\n===== 结果统计 =====")
        self.logger.log(f"总样本数: {len(df)}")
        self.logger.log("\n按游戏类型统计:")
        for game_type in df['game_type'].unique():
            type_df = df[df['game_type'] == game_type]
            self.logger.log(f"{game_type}: {len(type_df)} 个样本")
        
        self.logger.log("\n按shot设置统计:")
        for shot in df['shot_count'].unique():
            shot_df = df[df['shot_count'] == shot]
            self.logger.log(f"{shot}-shot: {len(shot_df)} 个样本")

    def save_results(self, filename: str):
        """保存结果到JSON文件"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def evaluate_image(self, image_path: Path, shot_count: int) -> Dict:
        """评估单个图像"""
        self.logger.log(f"评估: {image_path.name}")
        
        messages, true_result, game_type = self.build_prompt(image_path, shot_count)
        if not messages:
            return None
        
        start_time = time.time()
        response = self.call_model(messages)
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        if not response:
            return None
        
        response_text = response.choices[0].message.content
        prediction = self.parse_prediction(response_text)
        
        result = {
            "image": image_path.name,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text,
            "response_time": response_time
        }
        
        self.results.append(result)
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='三维物理直觉评估')
    parser.add_argument('--data_root', required=True, help='数据根目录')
    parser.add_argument('--output_dir', default='3d_results', help='输出目录')
    parser.add_argument('--num_images', type=int, default=20, help='测试图像数量')
    parser.add_argument('--shots', nargs='+', type=int, default=[0,1,2], help='Shot设置')
    parser.add_argument('--game_type', type=str, choices=['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support'],
                      help='要测试的游戏类型，如果不指定则测试所有类型')
    parser.add_argument('--model_name', type=str, default='deepseek-vl2', help='要使用的模型名称')
    parser.add_argument('--api_key', type=str, help='API密钥')
    parser.add_argument('--base_url', type=str, default='https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions', help='API基础URL')
        
    args = parser.parse_args()
        
    evaluator = PhysicalIntuition3DEvaluator(
        data_root=args.data_root,
        output_dir=args.output_dir,
        game_type=args.game_type,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    evaluator.run_evaluation(
        num_images=args.num_images,
        shot_counts=args.shots
    )

if __name__ == "__main__":
    main()