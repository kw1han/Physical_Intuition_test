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
import base64

class PhysicalIntuition3DEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "qwen-vl-plus",
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 output_dir: str = "3d_results"):
        """
        初始化三维物理直觉评估器
        
        Args:
            data_root: 三维数据根目录
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
            output_dir: 结果输出目录
        """
        # 初始化参数
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化API客户端
        self.client = OpenAI(
            api_key="sk-195225bba1f44e37aa394f1841d86a8e",
            base_url=base_url,
        )
        
        # 初始化结果记录
        self.results = []
        
        # 获取所有场景文件夹
        self.scenarios = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("Scene3D_")])
        
        # 三维场景类型映射
        self.scene_type_desc = {
            "Collision": "三维碰撞场景",
            "Gravity": "重力作用场景",
            "Fluid": "流体动力学场景",
            "Elasticity": "弹性力学场景"
        }

    def find_video_files(self, success_only: bool = None, scene_types: List[str] = None) -> List[Path]:
        """
        查找符合条件的视频文件
        
        Args:
            success_only: 是否筛选成功案例
            scene_types: 场景类型筛选
            
        Returns:
            符合条件的视频路径列表
        """
        valid_files = []
        
        for scene_dir in self.scenarios:
            for video_file in scene_dir.glob("*.mp4"):
                # 解析文件名格式: SceneType_ObjType_Success.mp4
                parts = video_file.stem.split('_')
                if len(parts) < 3:
                    continue
                
                scene_type = parts[0]
                is_success = parts[-1] == "Success"
                
                # 类型过滤
                if scene_types and scene_type not in scene_types:
                    continue
                
                # 成功状态过滤
                if success_only is not None and is_success != success_only:
                    continue
                
                valid_files.append(video_file)
        
        return valid_files

    def prepare_3d_examples(self, shot_count: int) -> List[Dict]:
        """
        准备三维示例视频
        
        Args:
            shot_count: 示例数量
            
        Returns:
            示例字典列表，包含视频路径和描述
        """
        examples = []
        success_files = self.find_video_files(success_only=True)
        failure_files = self.find_video_files(success_only=False)
        
        # 动态选择不同场景类型的示例
        selected_types = set()
        while len(examples) < shot_count:
            # 优先选择未出现的场景类型
            candidates = [f for f in (success_files + failure_files) 
                         if f.stem.split('_')[0] not in selected_types]
            
            if not candidates:
                candidates = success_files + failure_files
                
            selected = random.choice(candidates)
            scene_type = selected.stem.split('_')[0]
            is_success = "Success" in selected.stem
            
            examples.append({
                "video_path": str(selected),
                "is_success": is_success,
                "scene_type": scene_type,
                "description": f"在这个{self.scene_type_desc.get(scene_type, scene_type)}中，"
                              f"物体{'成功' if is_success else '未能'}完成预期运动。"
            })
            selected_types.add(scene_type)
        
        return examples[:shot_count]

    def encode_video_to_base64(self, video_path: str) -> str:
        """
        将视频文件编码为base64字符串
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            base64编码字符串
        """
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def build_3d_prompt(self, test_video: Path, shot_count: int) -> Tuple[List, bool, str]:
        """构建三维场景提示"""
        examples = self.prepare_3d_examples(shot_count)
        
        # 解析测试视频信息
        parts = test_video.stem.split('_')
        scene_type = parts[0]
        true_result = "Success" in test_video.stem
        
        messages = []
        
        # 系统提示
        system_msg = {
            "role": "system",
            "content": "你是一个三维物理直觉专家，请分析物体在三维空间中的运动规律，预测最终结果。"
        }
        messages.append(system_msg)
        
        # 添加示例
        for idx, ex in enumerate(examples):
            video_base64 = self.encode_video_to_base64(ex["video_path"])
            desc = ex["description"]
            
            example_content = [
                {"type": "text", "text": f"示例{idx+1}视频："},
                {"type": "video_url", 
                 "video_url": {
                     "url": f"data:video/mp4;base64,{video_base64}",
                     "duration": 1500  # 毫秒
                 }},
                {"type": "text", "text": desc}
            ]
            messages.append({"role": "user", "content": example_content})
            messages.append({"role": "assistant", "content": "理解示例中的三维物理规律。"})
        
        # 构建测试问题
        test_video_base64 = self.encode_video_to_base64(str(test_video))
        question = [
            {"type": "text", 
             "text": "请分析此三维物理场景，预测物体是否会完成预期运动？"
                     "结合示例中的规律，从空间关系、受力分析等角度详细解释。"},
            {"type": "video_url",
             "video_url": {
                 "url": f"data:video/mp4;base64,{test_video_base64}",
                 "duration": 1500
             }}
        ]
        messages.append({"role": "user", "content": question})
        
        return messages, true_result, scene_type

    def evaluate_3d_video(self, video_path: Path, shot_count: int) -> Dict:
        """评估单个三维视频"""
        print(f"评估: {video_path.name}")
        
        messages, true_result, scene_type = self.build_3d_prompt(video_path, shot_count)
        if not messages:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2  # 降低随机性
            )
        except Exception as e:
            print(f"API错误: {str(e)}")
            return None
        
        response_text = response.choices[0].message.content
        prediction = self.parse_3d_prediction(response_text)
        
        result = {
            "video": video_path.name,
            "scene_type": scene_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text
        }
        
        self.results.append(result)
        return result

    def parse_3d_prediction(self, text: str) -> bool:
        """解析三维预测结果"""
        text = text.lower()
        positive = any(kw in text for kw in ["会", "成功", "能", "是的"])
        negative = any(kw in text for kw in ["不会", "失败", "不能", "无法"])
        
        if positive and not negative:
            return True
        if negative and not positive:
            return False
        return random.choice([True, False])  # 模糊结果随机处理

    def run_3d_evaluation(self, num_videos=20, shot_counts=[0,1,2]):
        """运行三维评估"""
        selected_videos = self.select_diverse_videos(num_videos)
        
        for shot in shot_counts:
            print(f"\n=== 当前Shot设置: {shot} ===")
            for video in selected_videos:
                result = self.evaluate_3d_video(video, shot)
                if result:
                    print(f"结果: {result['correct']}")
                
            self.save_results(f"3d_results_{shot}shot.json")
        
        self.analyze_3d_results()

    def select_diverse_videos(self, num=20):
        """选择多样化的视频样本"""
        all_videos = self.find_video_files()
        type_counts = {}
        selected = []
        
        # 确保每个类型至少有2个样本
        for video in all_videos:
            scene_type = video.stem.split('_')[0]
            if type_counts.get(scene_type, 0) < 2:
                selected.append(video)
                type_counts[scene_type] = type_counts.get(scene_type, 0) + 1
            if len(selected) >= num:
                break
                
        # 填充剩余名额
        remaining = num - len(selected)
        if remaining > 0:
            selected += random.sample(all_videos, remaining)
            
        return selected[:num]

    def analyze_3d_results(self):
        """分析三维结果"""
        df = pd.DataFrame(self.results)
        
        # 总体统计
        accuracy = df.groupby(['scene_type', 'shot_count'])['correct'].mean().unstack()
        
        # 可视化
        plt.figure(figsize=(12,6))
        accuracy.plot(kind='bar')
        plt.title('三维物理直觉准确率分析')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / '3d_accuracy.png')
        
        # 保存详细结果
        df.to_csv(self.output_dir / '3d_results.csv', index=False)

def main_3d():
    parser = argparse.ArgumentParser(description='三维物理直觉评估')
    parser.add_argument('--data_root', required=True, help='三维数据根目录')
    parser.add_argument('--output_dir', default='3d_results', help='输出目录')
    parser.add_argument('--num_videos', type=int, default=20, help='测试视频数量')
    parser.add_argument('--shots', nargs='+', type=int, default=[0,1,2], help='Shot设置')
    
    args = parser.parse_args()
    
    evaluator = PhysicalIntuition3DEvaluator(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    evaluator.run_3d_evaluation(
        num_videos=args.num_videos,
        shot_counts=args.shots
    )

if __name__ == "__main__":
    main_3d()