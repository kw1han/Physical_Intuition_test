#!/usr/bin/env python3
import os
import sys
import argparse
import time
from openai import OpenAI
from pathlib import Path
import random
import base64
import json
from typing import List, Dict, Tuple, Optional, Union

def encode_image_to_base64(image_path):
    """将图像编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class PhysicalIntuitionOllamaEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "gemma3:27b",
                 base_url: str = "http://localhost:11434/v1",
                 output_dir: str = "results_ollama",
                 prompt_dir: str = "prompt1",
                 result_folder: str = None):
        """
        Initialize the physical intuition evaluator for Ollama models
        
        Args:
            data_root: Data root directory
            model_name: Model name to test (e.g., gemma3:27b, llama3.2-vision:11b, minicpm-v:8b)
            base_url: API base URL for Ollama
            output_dir: Results output directory
            prompt_dir: Directory containing scenario-specific prompts
            result_folder: Subfolder within output_dir to save results (defaults to timestamp if None)
        """
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create result subfolder with timestamp if not specified
        if result_folder is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_folder = f"{model_name.replace(':', '_')}_{timestamp}"
        
        self.result_folder = result_folder
        self.result_dir = self.output_dir / self.result_folder
        self.result_dir.mkdir(exist_ok=True)
        
        self.prompt_dir = Path(prompt_dir)
        
        # Initialize API client for Ollama's OpenAI-compatible API
        self.client = OpenAI(
            api_key="ollama",  # Required but unused by Ollama
            base_url=base_url,
        )
        
        # Check if this is a vision model
        self.is_vision_model = any(
            model_name in self.model_name.lower() 
            for model_name in ["vision", "llama3.2-vision", "minicpm-v", "gemma3"]
        )
        
        # Initialize results recording
        self.results = []
        
        # Get all subject folders
        self.subjects = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("Subj_")])
        
        # Game type mapping to English descriptions (for prompts)
        self.game_type_desc = {
            "Basic": "basic physical scenario",
            "Bridge": "bridge building scenario",
            "Catapult": "catapult scenario",
            "Chaining": "chain reaction scenario",
            "Falling": "object falling scenario",
            "Gap": "gap crossing scenario",
            "Launch": "launching scenario",
            "Prevention": "motion prevention scenario",
            "SeeSaw": "seesaw scenario",
            "Shafts": "shaft pathway scenario",
            "Table": "table scenario",
            "Unbox": "unboxing scenario",
            "Unsupport": "support removal scenario"
        }
        
        # Load scenario-specific prompts
        self.scenario_prompts = {}
        self._load_scenario_prompts()
    
    def _load_scenario_prompts(self):
        """Load scenario-specific prompts from the prompt directory"""
        print(f"正在从 {self.prompt_dir} 目录加载提示文件...")
        
        # 首先列出prompt_dir目录中的所有文件
        if self.prompt_dir.exists():
            print(f"找到提示目录: {self.prompt_dir}")
            existing_files = list(self.prompt_dir.glob("*.txt"))
            print(f"目录中的文件: {[f.name for f in existing_files]}")
        else:
            print(f"警告: 提示目录不存在: {self.prompt_dir}")
            
        for game_type in self.game_type_desc.keys():
            base_game_type = game_type.split('A')[0].split('B')[0]
            
            # 尝试多种可能的文件名格式
            possible_filenames = [
                f"{base_game_type.lower()}.txt",  # 全小写
                f"{base_game_type}.txt",          # 原始形式
                f"{base_game_type.upper()}.txt",  # 全大写
                f"{base_game_type.capitalize()}.txt"  # 首字母大写
            ]
            
            found = False
            for filename in possible_filenames:
                prompt_file = self.prompt_dir / filename
                if prompt_file.exists():
                    try:
                        with open(prompt_file, 'r') as f:
                            self.scenario_prompts[game_type] = f.read().strip()
                        print(f"已加载 {game_type} 的提示文件: {prompt_file}")
                        found = True
                        break
                    except Exception as e:
                        print(f"警告: 无法加载 {game_type} 的提示文件 {prompt_file}: {e}")
            
            if not found:
                print(f"警告: 未找到 {game_type} 的提示文件，尝试了以下文件名: {possible_filenames}")
    
    def find_trial_folders(self, success_only: bool = None, game_types: List[str] = None) -> List[Path]:
        """
        查找符合条件的试验文件夹
        
        Args:
            success_only: 如果为True，只返回成功的尝试；如果为False，只返回失败的尝试；如果为None，返回所有
            game_types: 要筛选的游戏类型列表
            
        Returns:
            符合条件的试验文件夹路径列表
        """
        all_trials = []
        
        for subject in self.subjects:
            for trial_folder in subject.iterdir():
                if not trial_folder.is_dir():
                    continue
                
                # 解析文件夹名称
                parts = trial_folder.name.split('_')
                if len(parts) < 5:
                    continue
                
                game_type = parts[0]
                is_success = parts[-1] == "True"
                
                # 过滤游戏类型
                if game_types and game_type not in game_types:
                    continue
                
                # 过滤成功/失败
                if success_only is not None and is_success != success_only:
                    continue
                
                all_trials.append(trial_folder)
        
        return all_trials
    
    def build_prompt(self, test_trial: Path) -> Tuple[List, bool, str]:
        """
        构建提示 (0-shot)
        
        Args:
            test_trial: 要测试的试验
            
        Returns:
            messages, true_result, game_type
        """
        # 尝试获取测试试验的第一帧
        try:
            first_frame = next(test_trial.glob("frame_0000.png"))
        except StopIteration:
            print(f"警告: 在 {test_trial} 中找不到第一帧图像，跳过此试验")
            return None, None, None
        
        # 获取真实结果
        true_result = test_trial.name.endswith("True")
        game_type = test_trial.name.split('_')[0]
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        messages = []
        
        # 系统提示 - 使用特定场景提示
        system_content = "You are an AI assistant with strong physical intuition. You need to predict whether the red ball will reach the green target area based on the physical scene image. Start your answer with a clear YES or NO before providing explanation."
        
        if game_type in self.scenario_prompts:
            system_content = self.scenario_prompts[game_type]
        elif base_game_type in self.scenario_prompts:
            system_content = self.scenario_prompts[base_game_type]
        
        messages.append({
            "role": "system", 
            "content": system_content
        })
        
        # 添加测试问题
        question_text = "Based on the initial scene image, will the red ball eventually reach the green target area? Start your answer with a clear YES or NO, then explain your reasoning process in detail, including the physical factors that affect the ball's movement."
        
        base64_first = encode_image_to_base64(str(first_frame))
        
        # 根据模型类型构建不同的消息格式
        if self.is_vision_model:
            # 对于支持视觉的模型，使用包含图像的消息格式
            question_content = [
                {"type": "text", "text": question_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_first}"}}
            ]
            messages.append({"role": "user", "content": question_content})
        else:
            # 对于仅文本模型，只包含文本描述
            messages.append({"role": "user", "content": f"[IMAGE: Initial scene showing a physics-based puzzle with a red ball and green target area]\n\n{question_text}"})
        
        return messages, true_result, game_type
    
    def call_model(self, messages: List[Dict], temperature: float = 0.0) -> Dict:
        """
        调用模型获取响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            模型响应
        """
        try:
            # 处理非视觉模型的消息
            if not self.is_vision_model:
                simplified_messages = []
                for msg in messages:
                    if "content" in msg and isinstance(msg["content"], list):
                        # 提取所有文本内容
                        text_content = " ".join([
                            item["text"] for item in msg["content"] 
                            if isinstance(item, dict) and item.get("type") == "text"
                        ])
                        simplified_msg = msg.copy()
                        simplified_msg["content"] = text_content
                        simplified_messages.append(simplified_msg)
                    else:
                        simplified_messages.append(msg)
                messages = simplified_messages
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            # 处理流式响应
            response_content = ""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    response_content += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            print("\n")  # 换行
            
            # 创建简化的响应对象
            class SimpleResponse:
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content
                    
                    def __init__(self, content):
                        self.message = self.Message(content)
                
                def __init__(self, content):
                    self.choices = [self.Choice(content)]
            
            return SimpleResponse(response_content)
            
        except Exception as e:
            print(f"\nAPI调用错误: {e}")
            
            # 尝试非流式模式
            try:
                print("尝试非流式模式...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                
                # 直接返回非流式响应
                return response
                
            except Exception as e2:
                print(f"重试失败: {e2}")
                return None
    
    def parse_prediction(self, response_text: str) -> bool:
        """
        从响应文本中解析预测结果
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            预测结果（True表示成功，False表示失败）
        """
        # 提取响应的前100个字符进行分析（通常YES/NO会在开头）
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
        
        # 如果以上都未匹配，默认返回False
        return False
    
    def evaluate_trial(self, trial_path: Path) -> Dict:
        """评估单个试验 (zero-shot)"""
        print(f"评估试验: {trial_path.name}")
        
        messages, true_result, game_type = self.build_prompt(trial_path)
        if messages is None:  # 检查是否应该跳过此试验
            return None
        
        response = self.call_model(messages)
        if not response:
            return {
                "trial": trial_path.name,
                "game_type": game_type,
                "true_result": true_result,
                "prediction": None,
                "correct": None,
                "response": None
            }
        
        response_text = response.choices[0].message.content
        prediction = self.parse_prediction(response_text)
        
        result = {
            "trial": trial_path.name,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text
        }
        
        # 保存结果
        self.results.append(result)
        
        return result
    
    def save_results(self, filename: str = None):
        """
        保存结果到JSON文件
        
        Args:
            filename: 自定义文件名
        """
        if filename is None:
            filename = f"results_{self.model_name.replace(':', '_')}.json"
            
        with open(self.result_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def select_diverse_trials(self, num_trials, balance_success=True):
        """
        选择多样化的试验样本，保证游戏类型和成功/失败案例的平衡
        
        Args:
            num_trials: 要选择的试验数量
            balance_success: 是否平衡成功和失败案例
            
        Returns:
            选择的试验路径列表
        """
        all_trials = self.find_trial_folders()
        
        if balance_success:
            # 按成功/失败分类
            success_trials = [t for t in all_trials if t.name.endswith("True")]
            failure_trials = [t for t in all_trials if t.name.endswith("False")]
            
            # 确定每类选择的数量
            num_per_category = num_trials // 2
            remainder = num_trials % 2
            
            # 打乱每类列表
            random.shuffle(success_trials)
            random.shuffle(failure_trials)
            
            # 从每类中选择相等数量
            selected_trials = success_trials[:num_per_category + remainder] + failure_trials[:num_per_category]
            
            # 再次打乱以随机化顺序
            random.shuffle(selected_trials)
            
            return selected_trials
        else:
            # 按游戏类型分组
            game_type_trials = {}
            for trial in all_trials:
                game_type = trial.name.split('_')[0]
                if game_type not in game_type_trials:
                    game_type_trials[game_type] = []
                game_type_trials[game_type].append(trial)
            
            # 从每种游戏类型中选择近似相等数量的试验
            selected_trials = []
            game_types = list(game_type_trials.keys())
            
            while len(selected_trials) < num_trials and game_types:
                for game_type in list(game_types):
                    if game_type_trials[game_type]:
                        trial = random.choice(game_type_trials[game_type])
                        game_type_trials[game_type].remove(trial)
                        selected_trials.append(trial)
                        
                        if len(selected_trials) >= num_trials:
                            break
                    else:
                        game_types.remove(game_type)
            
            return selected_trials
    
    def run_experiment(self, num_trials=10, temperature=0.0):
        """
        运行实验
        
        Args:
            num_trials: 试验数量
            temperature: 温度参数
        """
        # 重置结果
        self.results = []
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存实验参数
        experiment_params = {
            "experiment_type": "ollama_experiment",
            "num_trials": num_trials,
            "temperature": temperature,
            "model_name": self.model_name,
            "timestamp": timestamp
        }
        
        with open(self.result_dir / "experiment_params.json", 'w', encoding='utf-8') as f:
            json.dump(experiment_params, f, ensure_ascii=False, indent=2)
        
        # 选择多样化的试验样本
        print(f"选择 {num_trials} 个多样化试验样本...")
        test_trials = self.select_diverse_trials(num_trials)
        
        # 记录试验选择信息以便验证
        trial_info = [{"name": t.name, 
                       "game_type": t.name.split('_')[0], 
                       "is_success": t.name.endswith("True")} for t in test_trials]
        
        with open(self.result_dir / "selected_trials.json", 'w', encoding='utf-8') as f:
            json.dump(trial_info, f, ensure_ascii=False, indent=2)
        
        # 统计成功和失败案例数量
        success_count = sum(1 for t in test_trials if t.name.endswith("True"))
        failure_count = sum(1 for t in test_trials if t.name.endswith("False"))
        print(f"选择的试验: {success_count} 个成功案例, {failure_count} 个失败案例")
        
        # 打印游戏类型分布
        game_type_counts = {}
        for trial in test_trials:
            game_type = trial.name.split('_')[0]
            game_type_counts[game_type] = game_type_counts.get(game_type, 0) + 1
        
        print("游戏类型分布:")
        for game_type, count in game_type_counts.items():
            print(f"  {game_type}: {count}")
        
        # 评估每个试验
        for i, trial in enumerate(test_trials):
            print(f"\n===== 测试试验 {i+1}/{len(test_trials)}: {trial.name} =====")
            
            result = self.evaluate_trial(trial)
            if result:
                prediction_str = "成功" if result["prediction"] else "失败"
                actual_str = "成功" if result["true_result"] else "失败"
                correct_str = "正确" if result["correct"] else "错误"
                print(f"  结果: {correct_str} (预测: {prediction_str}, 实际: {actual_str})")
            
            # 保存中间结果
            self.save_results(f"intermediate_results_{i+1}.json")
            
            # 简短暂停避免API限制
            time.sleep(1)
        
        # 保存最终结果
        self.save_results("final_results.json")
        
        # 分析结果
        self.analyze_results()
        
        return self.results
    
    def analyze_results(self):
        """
        分析实验结果
        """
        if not self.results:
            print("没有结果可供分析")
            return
        
        # 计算总体准确率
        correct_count = sum(1 for r in self.results if r["correct"])
        total_count = len(self.results)
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        print("\n===== 实验结果分析 =====")
        print(f"总样本数: {total_count}")
        print(f"总体准确率: {overall_accuracy:.4f}")
        
        # 按游戏类型分析
        game_type_results = {}
        for result in self.results:
            game_type = result["game_type"]
            if game_type not in game_type_results:
                game_type_results[game_type] = {"correct": 0, "total": 0}
            
            game_type_results[game_type]["total"] += 1
            if result["correct"]:
                game_type_results[game_type]["correct"] += 1
        
        print("\n按游戏类型的准确率:")
        for game_type, data in game_type_results.items():
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            print(f"  {game_type}: {accuracy:.4f} ({data['correct']}/{data['total']})")
        
        # 成功案例和失败案例分析
        success_results = [r for r in self.results if r["true_result"]]
        failure_results = [r for r in self.results if not r["true_result"]]
        
        success_accuracy = sum(1 for r in success_results if r["correct"]) / len(success_results) if success_results else 0
        failure_accuracy = sum(1 for r in failure_results if r["correct"]) / len(failure_results) if failure_results else 0
        
        print("\n按案例类型的准确率:")
        print(f"  成功案例: {success_accuracy:.4f} ({sum(1 for r in success_results if r['correct'])}/{len(success_results)})")
        print(f"  失败案例: {failure_accuracy:.4f} ({sum(1 for r in failure_results if r['correct'])}/{len(failure_results)})")
        
        # 保存分析结果
        analysis_results = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_count,
            "game_type_accuracy": {gt: {"accuracy": data["correct"]/data["total"] if data["total"] > 0 else 0, 
                                       "correct": data["correct"], 
                                       "total": data["total"]} 
                                  for gt, data in game_type_results.items()},
            "success_case_accuracy": success_accuracy,
            "failure_case_accuracy": failure_accuracy
        }
        
        with open(self.result_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # 保存分析摘要到文本文件
        with open(self.result_dir / "analysis_summary.txt", 'w') as f:
            f.write("===== 实验结果分析 =====\n\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"总样本数: {total_count}\n")
            f.write(f"总体准确率: {overall_accuracy:.4f}\n\n")
            
            f.write("按游戏类型的准确率:\n")
            for game_type, data in game_type_results.items():
                accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
                f.write(f"  {game_type}: {accuracy:.4f} ({data['correct']}/{data['total']})\n")
            
            f.write("\n按案例类型的准确率:\n")
            f.write(f"  成功案例: {success_accuracy:.4f} ({sum(1 for r in success_results if r['correct'])}/{len(success_results)})\n")
            f.write(f"  失败案例: {failure_accuracy:.4f} ({sum(1 for r in failure_results if r['correct'])}/{len(failure_results)})\n")
        
        print("\n分析完成。结果已保存到输出目录。")

def main():
    """
    程序入口
    """
    parser = argparse.ArgumentParser(description="使用Ollama模型评估物理直觉")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--model_name", type=str, default="gemma3:27b", 
                        help="模型名称 (可选: gemma3:27b, llama3.2-vision:11b, minicpm-v:8b)")
    parser.add_argument("--base_url", type=str, default="http://localhost:11434/v1", help="Ollama API基础URL")
    parser.add_argument("--output_dir", type=str, default="results_ollama", help="输出目录")
    parser.add_argument("--prompt_dir", type=str, default="prompt1", help="提示目录")
    parser.add_argument("--result_folder", type=str, default=None, help="结果文件夹名称")
    parser.add_argument("--num_trials", type=int, default=10, help="试验次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
    
    # 创建评估器
    evaluator = PhysicalIntuitionOllamaEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        base_url=args.base_url,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        result_folder=args.result_folder
    )
    
    # 运行实验
    evaluator.run_experiment(num_trials=args.num_trials, temperature=args.temperature)

if __name__ == "__main__":
    main() 