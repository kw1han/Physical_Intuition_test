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
import csv

def encode_image_to_base64(image_path):
    """将图像编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class PhysicalIntuitionOllamaEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "gemma3:27b",
                 base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama",
                 output_dir: str = "results_ollama",
                 prompt_dir: str = "prompt1",
                 result_folder: str = None,
                 shot_condition: str = "0shot"):
        """
        Initialize the physical intuition evaluator for Ollama models
        
        Args:
            data_root: Data root directory
            model_name: Model name to test (e.g., gemma3:27b, llama3.2-vision:11b, minicpm-v:8b)
            base_url: API base URL for Ollama
            api_key: API key for service
            output_dir: Results output directory
            prompt_dir: Directory containing scenario-specific prompts
            result_folder: Subfolder within output_dir to save results (defaults to timestamp if None)
            shot_condition: Shot condition for the experiment (e.g., "0shot", "success_1shot", etc.)
        """
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shot_condition = shot_condition
        
        # Create result subfolder with timestamp if not specified
        if result_folder is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_folder = f"{model_name.replace(':', '_')}_{shot_condition}_{timestamp}"
        
        self.result_folder = result_folder
        self.result_dir = self.output_dir / self.result_folder
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompt_dir = Path(prompt_dir)
        
        # Initialize API client for Ollama's OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,  # Required but unused by Ollama
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
        
        # Initialize shot examples cache
        self.shot_examples_cache = {}
    
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
            # 修复游戏类型名称处理
            base_game_type = game_type
            if 'A' in game_type:
                base_game_type = game_type.split('A')[0]
            elif 'B' in game_type and not game_type.startswith(('Basic', 'Bridge')):
                base_game_type = game_type.split('B')[0]
            
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
                print(f"警告: 未找到 {game_type} 的提示文件 {base_game_type}.txt")
                # 尝试直接使用文件名匹配
                for existing_file in existing_files:
                    if existing_file.stem.lower() == base_game_type.lower():
                        try:
                            with open(existing_file, 'r') as f:
                                self.scenario_prompts[game_type] = f.read().strip()
                            print(f"已通过直接匹配加载 {game_type} 的提示文件: {existing_file}")
                            found = True
                            break
                        except Exception as e:
                            print(f"警告: 无法加载 {game_type} 的提示文件 {existing_file}: {e}")
                
                if not found:
                    print(f"警告: 所有尝试加载 {game_type} 的提示文件都失败了")
    
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
    
    def _get_best_1shot_strategy(self, game_type: str) -> str:
        """
        基于之前的1-shot实验结果确定最佳策略
        """
        one_shot_results = {
            "success": [],
            "failure": []
        }
        
        for result in self.results:
            if result["game_type"] == game_type and "_1shot" in result["shot_condition"]:
                if result["shot_condition"] == "success_1shot":
                    one_shot_results["success"].append(result["correct"])
                elif result["shot_condition"] == "failure_1shot":
                    one_shot_results["failure"].append(result["correct"])
        
        # 计算每种策略的准确率
        strategy_accuracy = {}
        for strategy, results in one_shot_results.items():
            if results:
                accuracy = sum(results) / len(results)
                strategy_accuracy[strategy] = accuracy
        
        if not strategy_accuracy:
            return "success"
        
        return max(strategy_accuracy.items(), key=lambda x: x[1])[0]

    def _get_best_2shot_strategy(self, game_type: str, best_1shot: str) -> str:
        """
        基于最佳1-shot策略确定2-shot的第二个示例类型
        """
        two_shot_results = {
            "success": [],
            "failure": []
        }
        
        for result in self.results:
            if result["game_type"] == game_type and "_2shot" in result["shot_condition"]:
                if best_1shot == "success":
                    if result["shot_condition"] == "success_success_2shot":
                        two_shot_results["success"].append(result["correct"])
                    elif result["shot_condition"] == "success_failure_2shot":
                        two_shot_results["failure"].append(result["correct"])
                elif best_1shot == "failure":
                    if result["shot_condition"] == "failure_success_2shot":
                        two_shot_results["success"].append(result["correct"])
                    elif result["shot_condition"] == "failure_failure_2shot":
                        two_shot_results["failure"].append(result["correct"])
        
        strategy_accuracy = {}
        for strategy, results in two_shot_results.items():
            if results:
                accuracy = sum(results) / len(results)
                strategy_accuracy[strategy] = accuracy
        
        if not strategy_accuracy:
            return "failure" if best_1shot == "success" else "success"
        
        return max(strategy_accuracy.items(), key=lambda x: x[1])[0]

    def _get_shot_examples(self, game_type: str, test_trial: Path, all_trials: List[Path]) -> List[Dict]:
        """获取shot示例"""
        if self.shot_condition == "0shot":
            return []
        
        available_trials = [
            t for t in all_trials 
            if t.name.split('_')[0] == game_type and t != test_trial
        ]
        
        success_trials = [t for t in available_trials if t.name.endswith("True")]
        failure_trials = [t for t in available_trials if t.name.endswith("False")]
        
        examples = []
        used_trials = set()
        
        # 1-shot 情况：直接使用指定的条件
        if "_1shot" in self.shot_condition:
            is_success = "success" in self.shot_condition
            available = success_trials if is_success else failure_trials
            if available:
                example = random.choice(available)
                examples.append({
                    "trial_path": example,
                    "true_result": is_success,
                    "obj": example.name.split('_')[3]
                })
            return examples
        
        # 获取1-shot的结果
        one_shot_condition = "success_1shot" if self.shot_condition.startswith("success") else "failure_1shot"
        one_shot_results = [r for r in self.results if r["game_type"] == game_type and r["shot_condition"] == one_shot_condition]
        
        # 使用1-shot的示例作为第一个示例
        if one_shot_results:
            # 找到对应的trial
            one_shot_trial = next(
                (t for t in available_trials if t.name == one_shot_results[-1]["trial"]),
                None
            )
            if one_shot_trial:
                examples.append({
                    "trial_path": one_shot_trial,
                    "true_result": one_shot_trial.name.endswith("True"),
                    "obj": one_shot_trial.name.split('_')[3]
                })
                used_trials.add(one_shot_trial)
        
        # 2-shot 情况：使用1-shot的结果加上新的示例
        if "_2shot" in self.shot_condition:
            # 获取第二个shot的条件
            second_shot_type = self.shot_condition.split('_')[1]
            is_second_success = second_shot_type == "success"
            
            # 添加第二个示例
            available = [
                t for t in (success_trials if is_second_success else failure_trials)
                if t not in used_trials
            ]
            if available:
                second_example = random.choice(available)
                examples.append({
                    "trial_path": second_example,
                    "true_result": is_second_success,
                    "obj": second_example.name.split('_')[3]
                })
            return examples
        
        # 3-shot 情况：使用2-shot的结果加上新的示例
        if "best2shot" in self.shot_condition:
            # 获取2-shot的结果
            two_shot_condition = f"{one_shot_condition.split('_')[0]}_{self.shot_condition.split('_')[-1]}_2shot"
            two_shot_results = [r for r in self.results if r["game_type"] == game_type and r["shot_condition"] == two_shot_condition]
            
            if two_shot_results:
                # 找到对应的第二个trial
                two_shot_trial = next(
                    (t for t in available_trials if t.name == two_shot_results[-1]["trial"]),
                    None
                )
                if two_shot_trial and two_shot_trial not in used_trials:
                    examples.append({
                        "trial_path": two_shot_trial,
                        "true_result": two_shot_trial.name.endswith("True"),
                        "obj": two_shot_trial.name.split('_')[3]
                    })
                    used_trials.add(two_shot_trial)
            
            # 添加第三个示例
            is_third_success = "success" in self.shot_condition.split('_')[-1]
            available = [
                t for t in (success_trials if is_third_success else failure_trials)
                if t not in used_trials
            ]
            if available:
                third_example = random.choice(available)
                examples.append({
                    "trial_path": third_example,
                    "true_result": is_third_success,
                    "obj": third_example.name.split('_')[3]
                })
        
        return examples
    
    def build_prompt(self, test_trial: Path, all_trials: List[Path]) -> Tuple[List, bool, str]:
        """构建提示 (包含shot示例)"""
        # 尝试获取测试试验的第一帧
        try:
            first_frame = next(test_trial.glob("frame_0000.png"))
        except StopIteration:
            print(f"警告: 在 {test_trial} 中找不到第一帧图像，跳过此试验")
            return None, None, None
        
        # 获取真实结果和游戏类型
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
            
        # 添加shot示例
        if self.shot_condition != "0shot":
            shot_examples = self._get_shot_examples(game_type, test_trial, all_trials)
            if shot_examples:
                system_content += "\n\nHere are some example scenarios and their outcomes:\n"
                for i, example in enumerate(shot_examples, 1):
                    example_path = example["trial_path"]
                    example_frame = next(example_path.glob("frame_0000.png"))
                    example_result = "YES" if example["true_result"] else "NO"
                    
                    system_content += f"\nExample {i}:\n"
                    if self.is_vision_model:
                        example_base64 = encode_image_to_base64(str(example_frame))
                        system_content += f"[Example Image {i}]\n"
                    system_content += f"Q: Will the red ball reach the green target area in this physical scenario?\n"
                    system_content += f"A: {example_result}. [Detailed explanation would be provided here]\n"
        
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
            
            # 如果有shot示例，添加示例图像
            if self.shot_condition != "0shot":
                shot_examples = self._get_shot_examples(game_type, test_trial, all_trials)
                for example in shot_examples:
                    example_frame = next(example["trial_path"].glob("frame_0000.png"))
                    example_base64 = encode_image_to_base64(str(example_frame))
                    question_content.insert(-1, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{example_base64}"}
                    })
            
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
        start_time = time.time()
        response_time = None
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
            response_time = time.time() - start_time
            
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
                    self.response_time = None  # 添加响应时间字段
            
            response_obj = SimpleResponse(response_content)
            response_obj.response_time = response_time  # 设置响应时间
            return response_obj
            
        except Exception as e:
            print(f"\nAPI调用错误: {e}")
            
            # 尝试非流式模式
            try:
                print("尝试非流式模式...")
                non_stream_start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                response_time = time.time() - non_stream_start
                response.response_time = response_time  # 为非流式响应也添加时间
                return response
                
            except Exception as e2:
                print(f"重试失败: {e2}")
                return None
        finally:
            if response_time is not None:
                print(f"响应时间: {response_time:.2f}秒")
    
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
        
        messages, true_result, game_type = self.build_prompt(trial_path, self.find_trial_folders())
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
                "response": None,
                "response_time": None
            }
        
        response_text = response.choices[0].message.content
        prediction = self.parse_prediction(response_text)
        
        result = {
            "trial": trial_path.name,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text,
            "response_time": getattr(response, 'response_time', None)  # 添加响应时间
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
    
    def save_results_csv(self, filename: str = None):
        """
        保存结果到CSV文件
        
        Args:
            filename: 自定义文件名
        """
        if filename is None:
            filename = f"results_{self.model_name.replace(':', '_')}.csv"
        
        with open(self.result_dir / filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                "trial_name", "game_type", "true_result", "prediction", 
                "correct", "shot_condition", "response_time", "response"  # 添加response_time字段
            ])
            # 写入数据
            for result in self.results:
                writer.writerow([
                    result["trial"],
                    result["game_type"],
                    result["true_result"],
                    result["prediction"],
                    result["correct"],
                    self.shot_condition,
                    result.get("response_time", None),  # 添加响应时间
                    result["response"]
                ])
    
    def select_diverse_trials(self, num_trials, balance_success=True, all_samples=False):
        """选择多样化的试验样本"""
        all_trials = self.find_trial_folders()
        
        if all_samples:
            return all_trials
        
            # 按游戏类型分组
            game_type_trials = {}
            for trial in all_trials:
                game_type = trial.name.split('_')[0]
                if game_type not in game_type_trials:
                game_type_trials[game_type] = {
                    'success': [],
                    'failure': []
                }
            # 按成功/失败分类
            if trial.name.endswith("True"):
                game_type_trials[game_type]['success'].append(trial)
            else:
                game_type_trials[game_type]['failure'].append(trial)
        
            selected_trials = []
        # 从每个游戏类型中选择固定数量的样本
        samples_per_type = num_trials // len(game_type_trials)
        if samples_per_type == 0:
            samples_per_type = 1
        
        success_per_type = samples_per_type // 4  # 保持1:3比例
        failure_per_type = samples_per_type - success_per_type
        
        for game_type, trials in game_type_trials.items():
            # 随机选择成功案例
            if trials['success']:
                random.shuffle(trials['success'])
                selected_trials.extend(trials['success'][:success_per_type])
            
            # 随机选择失败案例
            if trials['failure']:
                random.shuffle(trials['failure'])
                selected_trials.extend(trials['failure'][:failure_per_type])
        
        # 打乱最终的选择顺序
        random.shuffle(selected_trials)
            return selected_trials
    
    def get_trials_for_game_type(self, game_type: str, num_trials: int = 52) -> List[Path]:
        """获取指定游戏类型的试验样本"""
        # 获取该游戏类型的所有样本
        all_trials = [t for t in self.find_trial_folders() if t.name.split('_')[0] == game_type]
        
        # 如果样本数量不足，返回全部样本
        if len(all_trials) <= num_trials:
            return all_trials
        
        # 随机选择指定数量的样本
        return random.sample(all_trials, num_trials)

    def test_single_trial_all_shots(self, trial: Path, game_type: str) -> List[Dict]:
        """测试单个样本在所有shot条件下的表现"""
        results = []
        
        # 获取所有可用的样本（用于选择shot示例）
        all_trials = self.find_trial_folders()
        
        # 缓存第一帧图像
        try:
            first_frame = next(trial.glob("frame_0000.png"))
            first_frame_base64 = encode_image_to_base64(str(first_frame))
        except StopIteration:
            print(f"警告: 在 {trial} 中找不到第一帧图像")
            return results
        
        # 获取真实结果
        true_result = trial.name.endswith("True")
        
        # 对每个shot条件进行测试
        for shot_condition in [
            "0shot",
            "success_1shot", "failure_1shot",
            "success_success_2shot", "success_failure_2shot",
            "failure_success_2shot", "failure_failure_2shot",
            "best2shot_success_3shot", "best2shot_failure_3shot"
        ]:
            print(f"\nTesting {shot_condition}...")
            
            # 设置当前的shot条件
            self.shot_condition = shot_condition
            
            # 构建提示
            messages, _, _ = self.build_prompt(trial, all_trials)
            if messages is None:
                continue
            
            # 调用模型
            response = self.call_model(messages)
            if not response:
                continue
            
            response_text = response.choices[0].message.content
            prediction = self.parse_prediction(response_text)
            
            # 记录结果
            result = {
                "trial": trial.name,
                "game_type": game_type,
                "shot_condition": shot_condition,
                "true_result": true_result,
                "prediction": prediction,
                "correct": prediction == true_result,
                "response": response_text
            }
            
            results.append(result)
            
            # 保存中间结果
            self.results.extend([result])
            self.save_results(f"intermediate_results_{trial.name}_{shot_condition}.json")
            self.save_results_csv(f"intermediate_results_{trial.name}_{shot_condition}.csv")
            
            # 短暂暂停避免API限制
            time.sleep(5)
        
        return results

    def run_experiment(self, num_trials=52, temperature=0.0, all_samples=False, game_type=None):
        """运行实验的主函数"""
        print("\n开始实验...")
        print(f"模型: {self.model_name}")
        print(f"Shot条件将依次测试: {self.shot_condition}")
        
        # 如果指定了游戏类型，只测试该类型
        if game_type:
            game_types = [game_type]
        else:
            game_types = list(self.game_type_desc.keys())
            random.shuffle(game_types)  # 随机打乱游戏类型顺序
        
        total_results = []
        
        # 对每个游戏类型进行测试
        for game_type in game_types:
            print(f"\n===== 测试游戏类型: {game_type} =====")
            
            # 获取该游戏类型的试验样本
            trials = self.get_trials_for_game_type(game_type, num_trials)
            print(f"选择了 {len(trials)} 个样本进行测试")
            
            # 测试每个样本
            for i, trial in enumerate(trials):
                print(f"\n----- 测试样本 {i+1}/{len(trials)}: {trial.name} -----")
                
                # 测试该样本在所有shot条件下的表现
                results = self.test_single_trial_all_shots(trial, game_type)
                total_results.extend(results)
                
                # 在不同样本之间暂停较长时间
                time.sleep(30)
            
            # 在不同游戏类型之间暂停更长时间
            time.sleep(300)
        
        # 保存最终结果
        self.save_results("final_results.json")
        self.save_results_csv("final_results.csv")
        
        # 分析结果
        self.analyze_results()
        
        return total_results
    
    def analyze_results(self):
        """
        分析实验结果
        """
        if not self.results:
            print("没有结果可供分析")
            return
        
        # 计算总体准确率和平均响应时间
        correct_count = sum(1 for r in self.results if r["correct"])
        total_count = len(self.results)
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        # 计算平均响应时间
        response_times = [r["response_time"] for r in self.results if r.get("response_time") is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        print(f"\n===== 实验结果分析 ({self.shot_condition}) =====")
        print(f"总样本数: {total_count}")
        print(f"总体准确率: {overall_accuracy:.4f}")
        if avg_response_time is not None:
            print(f"平均响应时间: {avg_response_time:.2f}秒")
        
        # 按游戏类型分析
        game_type_results = {}
        for result in self.results:
            game_type = result["game_type"]
            if game_type not in game_type_results:
                game_type_results[game_type] = {
                    "correct": 0, 
                    "total": 0,
                    "response_times": []
                }
            
            game_type_results[game_type]["total"] += 1
            if result["correct"]:
                game_type_results[game_type]["correct"] += 1
            if result.get("response_time") is not None:
                game_type_results[game_type]["response_times"].append(result["response_time"])
        
        print("\n按游戏类型的准确率和响应时间:")
        for game_type, data in game_type_results.items():
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            avg_time = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else None
            time_str = f", 平均响应时间: {avg_time:.2f}秒" if avg_time is not None else ""
            print(f"  {game_type}: 准确率 {accuracy:.4f} ({data['correct']}/{data['total']}){time_str}")
        
        # 成功案例和失败案例分析
        success_results = [r for r in self.results if r["true_result"]]
        failure_results = [r for r in self.results if not r["true_result"]]
        
        success_accuracy = sum(1 for r in success_results if r["correct"]) / len(success_results) if success_results else 0
        failure_accuracy = sum(1 for r in failure_results if r["correct"]) / len(failure_results) if failure_results else 0
        
        # 计算成功和失败案例的平均响应时间
        success_times = [r["response_time"] for r in success_results if r.get("response_time") is not None]
        failure_times = [r["response_time"] for r in failure_results if r.get("response_time") is not None]
        
        avg_success_time = sum(success_times) / len(success_times) if success_times else None
        avg_failure_time = sum(failure_times) / len(failure_times) if failure_times else None
        
        print("\n按案例类型的准确率和响应时间:")
        success_time_str = f", 平均响应时间: {avg_success_time:.2f}秒" if avg_success_time is not None else ""
        failure_time_str = f", 平均响应时间: {avg_failure_time:.2f}秒" if avg_failure_time is not None else ""
        print(f"  成功案例: 准确率 {success_accuracy:.4f} ({sum(1 for r in success_results if r['correct'])}/{len(success_results)}){success_time_str}")
        print(f"  失败案例: 准确率 {failure_accuracy:.4f} ({sum(1 for r in failure_results if r['correct'])}/{len(failure_results)}){failure_time_str}")
        
        # 保存分析结果
        analysis_results = {
            "shot_condition": self.shot_condition,
            "overall_accuracy": overall_accuracy,
            "average_response_time": avg_response_time,
            "total_samples": total_count,
            "game_type_analysis": {
                gt: {
                    "accuracy": data["correct"]/data["total"] if data["total"] > 0 else 0,
                                       "correct": data["correct"], 
                    "total": data["total"],
                    "average_response_time": sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else None
                } for gt, data in game_type_results.items()
            },
            "success_case_analysis": {
                "accuracy": success_accuracy,
                "average_response_time": avg_success_time
            },
            "failure_case_analysis": {
                "accuracy": failure_accuracy,
                "average_response_time": avg_failure_time
            }
        }
        
        with open(self.result_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # 保存分析摘要到文本文件
        with open(self.result_dir / "analysis_summary.txt", 'w') as f:
            f.write(f"===== 实验结果分析 ({self.shot_condition}) =====\n\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"总样本数: {total_count}\n")
            f.write(f"总体准确率: {overall_accuracy:.4f}\n")
            if avg_response_time is not None:
                f.write(f"平均响应时间: {avg_response_time:.2f}秒\n")
            
            f.write("\n按游戏类型的准确率和响应时间:\n")
            for game_type, data in game_type_results.items():
                accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
                avg_time = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else None
                time_str = f", 平均响应时间: {avg_time:.2f}秒" if avg_time is not None else ""
                f.write(f"  {game_type}: 准确率 {accuracy:.4f} ({data['correct']}/{data['total']}){time_str}\n")
            
            f.write("\n按案例类型的准确率和响应时间:\n")
            f.write(f"  成功案例: 准确率 {success_accuracy:.4f} ({sum(1 for r in success_results if r['correct'])}/{len(success_results)}){success_time_str}\n")
            f.write(f"  失败案例: 准确率 {failure_accuracy:.4f} ({sum(1 for r in failure_results if r['correct'])}/{len(failure_results)}){failure_time_str}\n")
        
        print("\n分析完成。结果已保存到输出目录。")

    def run_single_trial(self, trial_name: str, game_type: str, shot_condition: str):
        """测试单个样本在特定shot条件下的表现"""
        # 获取所有可用的样本（用于选择shot示例）
        all_trials = self.find_trial_folders()
        
        # 找到指定的测试样本
        test_trial = next(t for t in all_trials if t.name == trial_name)
        
        # 评估该样本
        result = self.evaluate_trial(test_trial)
        
        # 保存结果
        result_dir = Path(self.output_dir) / self.model_name.replace(':', '_') / game_type / trial_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV格式
        with open(result_dir / f"{shot_condition}_result.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_name", "game_type", "shot_condition", 
                "true_result", "prediction", "correct", "response"
            ])
            writer.writerow([
                trial_name, game_type, shot_condition,
                result["true_result"], result["prediction"], 
                result["correct"], result["response"]
            ])

def analyze_results_by_trial(base_dir: Path):
    """分析每个样本在不同shot条件下的表现"""
    results = {}
    
    # 遍历所有游戏类型
    for game_dir in base_dir.glob('*'):
        game_type = game_dir.name
        results[game_type] = {}
        
        # 遍历该游戏类型的所有样本
        for trial_dir in game_dir.glob('*'):
            trial_name = trial_dir.name
            results[game_type][trial_name] = {
                'shot_conditions': {},
                'improvement': None
            }
            
            # 读取该样本在所有shot条件下的结果
            for result_file in trial_dir.glob('*_result.csv'):
                shot_condition = result_file.stem.replace('_result', '')
                with open(result_file) as f:
                    reader = csv.DictReader(f)
                    result = next(reader)
                    results[game_type][trial_name]['shot_conditions'][shot_condition] = result
            
            # 计算相对于0-shot的改进
            zero_shot = results[game_type][trial_name]['shot_conditions']['0shot']['correct']
            best_shot = max(
                (r['correct'] for c, r in results[game_type][trial_name]['shot_conditions'].items() 
                if c != '0shot'),
                default=zero_shot
            )
            results[game_type][trial_name]['improvement'] = best_shot - zero_shot
    
    # 保存分析结果
    with open(base_dir / 'trial_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """
    程序入口
    """
    parser = argparse.ArgumentParser(description="使用Ollama模型评估物理直觉")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--model_name", type=str, default="gemma3:27b", 
                        help="模型名称 (可选: gemma3:27b, llama3.2-vision:11b, minicpm-v:8b)")
    parser.add_argument("--base_url", type=str, default="http://localhost:11434/v1", help="Ollama API基础URL")
    parser.add_argument("--api_key", type=str, default="ollama", help="API密钥")
    parser.add_argument("--output_dir", type=str, default="results_ollama", help="输出目录")
    parser.add_argument("--prompt_dir", type=str, default="prompt1", help="提示目录")
    parser.add_argument("--result_folder", type=str, default=None, help="结果文件夹名称")
    parser.add_argument("--num_trials", type=int, default=10, help="试验次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    parser.add_argument("--all_samples", action="store_true", help="测试所有样本")
    parser.add_argument("--game_type", type=str, default=None, help="指定要测试的游戏类型")
    parser.add_argument("--shot_condition", type=str, default="0shot",
                        choices=["0shot", 
                                "success_1shot", "failure_1shot",
                                "success_success_2shot", "success_failure_2shot",
                                "failure_success_2shot", "failure_failure_2shot",
                                "best2shot_success_3shot", "best2shot_failure_3shot"],
                        help="Shot实验条件")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
    
    # 创建评估器
    evaluator = PhysicalIntuitionOllamaEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        result_folder=args.result_folder,
        shot_condition=args.shot_condition
    )
    
    # 运行实验
    evaluator.run_experiment(
        num_trials=args.num_trials, 
        temperature=args.temperature, 
        all_samples=args.all_samples,
        game_type=args.game_type
    )

if __name__ == "__main__":
    main() 