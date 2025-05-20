import os
import json
import random
from pathlib import Path
import argparse
from openai import OpenAI
from typing import List, Dict, Tuple, Optional, Union
import time
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np
import re

def encode_image_to_base64(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class PhysicalIntuitionEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "qwen2.5-omni-7b",
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 output_dir: str = "results",
                 prompt_dir: str = "prompt",
                 result_folder: str = None):
        """
        Initialize the physical intuition evaluator
        
        Args:
            data_root: Data root directory
            model_name: Model name to test
            api_key: API key
            base_url: API base URL
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
            result_folder = f"experiment_{timestamp}"
        
        self.result_folder = result_folder
        self.result_dir = self.output_dir / self.result_folder
        self.result_dir.mkdir(exist_ok=True)
        
        self.prompt_dir = Path(prompt_dir)
        
        # Initialize API client
        self.client = OpenAI(
            api_key="sk-195225bba1f44e37aa394f1841d86a8e",
            base_url=base_url,
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
        
        # 添加一个集合来跟踪所有已使用的示例
        self.used_examples = set()
    
    def _load_scenario_prompts(self):
        """Load scenario-specific prompts from the prompt directory"""
        for game_type in self.game_type_desc.keys():
            # 移除可能的A或B后缀，只保留基本游戏类型名称
            base_game_type = game_type.split('A')[0].split('B')[0]
            prompt_file = self.prompt_dir / f"{base_game_type}.txt"
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r') as f:
                        self.scenario_prompts[game_type] = f.read().strip()
                except Exception as e:
                    print(f"Warning: Could not load prompt for {game_type}: {e}")
            else:
                print(f"Warning: No prompt file found for {game_type}")
    
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
    
    def get_all_game_types(self):
        """返回数据集中所有游戏类型"""
        all_trials = self.find_trial_folders()
        return set(t.name.split('_')[0] for t in all_trials)
    
    def build_prompt(self, test_trial: Path, shot_count: int) -> Tuple[List, bool, str]:
        """构建提示"""
        examples = []
        if shot_count > 0:
            # 移除对prepare_shot_examples的调用
            # examples = self.prepare_shot_examples(shot_count)
            print(f"错误: build_prompt不再支持基础的shot_count方法，请使用控制实验功能")
            return None, None, None
        
        # 尝试获取测试试验的第一帧
        try:
            first_frame = next(test_trial.glob("frame_0000.png"))
        except StopIteration:
            print(f"警告: 在 {test_trial} 中找不到第一帧图像，跳过此试验")
            return None, None, None  # 返回None表示跳过此试验
        
        # 获取真实结果
        true_result = test_trial.name.endswith("True")
        game_type = test_trial.name.split('_')[0]
        
        messages = []
        
        # 系统提示
        system_message = {
            "role": "system", 
            "content": "You are an AI assistant with strong physical intuition. You need to predict whether the red ball will reach the green target area based on the physical scene image. Start your answer with a clear YES or NO before providing explanation."
        }
        messages.append(system_message)
        
        # 添加few-shot示例
        for i, example in enumerate(examples):
            base64_initial = encode_image_to_base64(example['initial_frame'])
            base64_final = encode_image_to_base64(example['final_frame'])
            
            example_content = [
                {"type": "text", "text": f"Initial state of Example {i+1}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_initial}"}}
            ]
            
            # 添加中间帧（如果有）
            if 'middle_frame1' in example and 'middle_frame2' in example:
                base64_middle1 = encode_image_to_base64(example['middle_frame1'])
                base64_middle2 = encode_image_to_base64(example['middle_frame2'])
                
                example_content.append({"type": "text", "text": f"Process frame 1 of Example {i+1}:"})
                example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_middle1}"}})
                example_content.append({"type": "text", "text": f"Process frame 2 of Example {i+1}:"})
                example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_middle2}"}})
            
            example_content.extend([
                {"type": "text", "text": f"Final state of Example {i+1}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_final}"}},
                {"type": "text", "text": example["description"]}
            ])
            
            messages.append({"role": "user", "content": example_content})
            messages.append({"role": "assistant", "content": f"{'YES' if example['is_success'] else 'NO'}, I understand. In this {self.game_type_desc.get(example['game_type'], example['game_type'])} scenario, the red ball {'successfully reached' if example['is_success'] else 'failed to reach'} the green target area."})
        
        # 添加测试问题
        question_text = "Based on the initial scene image, will the red ball eventually reach the green target area? Start your answer with a clear YES or NO, then explain your reasoning process in detail, including the physical factors that affect the ball's movement."
        
        if shot_count > 0:
            question_text = "Based on the previous examples and your physical knowledge, " + question_text
        
        # 如果有3个示例，添加反思提示
        if shot_count >= 3:
            question_text += " Before answering, please reflect on the physical principles in the previous examples and summarize the key factors that may influence success or failure."
        
        base64_first = encode_image_to_base64(str(first_frame))
        question_content = [
            {"type": "text", "text": question_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_first}"}}
        ]
        
        messages.append({"role": "user", "content": question_content})
        
        return messages, true_result, game_type
    
    def call_model(self, messages, temperature=0.0):
        """
        调用OpenAI API并获取响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数，控制随机性
            
        Returns:
            tuple: (预测结果（True/False/None），解释文本)
        """
        if not isinstance(messages, list):
            print("错误：消息格式无效，预期为列表")
            return None, "消息格式无效"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=False  # 不使用流式响应
            )
            
            # 获取模型响应
            model_response = response.choices[0].message.content
            
            # 解析预测结果和提取解释
            prediction = self.parse_prediction(model_response)
            explanation = self.extract_explanation(model_response)
            
            return prediction, explanation
            
        except Exception as e:
            print(f"API调用错误：{str(e)}")
            return None, f"API错误：{str(e)}"
            
    def parse_prediction(self, response):
        """
        从模型响应中解析预测结果
        
        Args:
            response: 模型响应文本
            
        Returns:
            bool: True表示成功，False表示失败，None表示无法确定
        """
        # 成功模式
        success_patterns = [
            r'\b(success|succeed|successful|will\s+succeed|can\s+succeed)\b',
            r'\b(成功|会成功|可以成功)\b',
            r'\b(yes)\b',
            r'\b(是)\b'
        ]
        
        # 失败模式
        failure_patterns = [
            r'\b(fail|fails|failure|will\s+fail|cannot\s+succeed|won\'t\s+succeed)\b',
            r'\b(失败|会失败|不会成功)\b',
            r'\b(no)\b',
            r'\b(否)\b'
        ]
        
        # 检查成功模式
        success_match = any(re.search(pattern, response, re.IGNORECASE) for pattern in success_patterns)
        
        # 检查失败模式
        failure_match = any(re.search(pattern, response, re.IGNORECASE) for pattern in failure_patterns)
        
        # 处理冲突情况
        if success_match and failure_match:
            # 统计"success"和"fail"的出现次数
            success_count = len(re.findall(r'\b(success|succeed|successful)\b', response, re.IGNORECASE))
            failure_count = len(re.findall(r'\b(fail|fails|failure)\b', response, re.IGNORECASE))
            
            if success_count > failure_count:
                return True
            elif failure_count > success_count:
        return False
            else:
                print(f"警告：无法确定预测结果，成功和失败模式都匹配，次数相同")
                return None
        elif success_match:
            return True
        elif failure_match:
            return False
        else:
            print(f"警告：无法确定预测结果，没有匹配到成功或失败模式")
            return None
            
    def extract_explanation(self, response):
        """
        从模型响应中提取解释部分
        
        Args:
            response: 模型响应文本
            
        Returns:
            str: 解释文本
        """
        # 简单实现：返回完整响应
        # 如果需要更复杂的提取逻辑，可以在这里实现
        return response
    
    def evaluate_trial(self, trial_path: Path, shot_count: int) -> Dict:
        """评估单个试验"""
        print(f"评估: {trial_path.name}, {shot_count}-shot")
        
        messages, true_result, game_type = self.build_prompt(trial_path, shot_count)
        if messages is None:  # 检查是否应该跳过此试验
            return None
        
        response = self.call_model(messages)
        if not response:
            return {
                "trial": trial_path.name,
                "shot_count": shot_count,
                "game_type": game_type,
                "true_result": true_result,
                "prediction": None,
                "correct": None,
                "response": None,
                "confidence": None
            }
        
        response_text = response.choices[0].message.content
        prediction = self.parse_prediction(response_text)
        
        # 简单计算置信度（基于关键词出现的次数和位置）
        confidence = None  # 可以根据需要实现置信度计算
        
        result = {
            "trial": trial_path.name,
            "shot_count": shot_count,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result,
            "response": response_text,
            "confidence": confidence
        }
        
        # 保存结果
        self.results.append(result)
        
        return result
    
    def save_results(self, filename: str = None):
        """
        Save results to JSON file
        
        Args:
            filename: Custom filename (defaults to "results_{timestamp}.json" if None)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
            
        with open(self.result_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def select_diverse_trials(self, num_trials, balance_success=False):
        """
        Select a diverse set of trials considering game types and success/failure cases.
        
        Args:
            num_trials: Number of trials to select
            balance_success: Whether to balance success and failure cases (forces equal selection)
            
        Returns:
            List of selected trial objects
        """
        all_trials = self.find_trial_folders()
        
        if balance_success:
            # Separate trials by success/failure
            success_trials = [t for t in all_trials if t.name.endswith("True")]
            failure_trials = [t for t in all_trials if t.name.endswith("False")]
            
            # Determine number to select from each category
            num_per_category = num_trials // 2
            remainder = num_trials % 2
            
            # Shuffle each category
            random.shuffle(success_trials)
            random.shuffle(failure_trials)
            
            # Select equal numbers from each category (plus remainder)
            selected_trials = success_trials[:num_per_category + remainder] + failure_trials[:num_per_category]
            
            # Shuffle the combined selection to randomize order
            random.shuffle(selected_trials)
            
            return selected_trials
        else:
            # Original method - maintain for backward compatibility
            # Group trials by game type
            game_type_trials = {}
            for trial in all_trials:
                game_type = trial.name.split('_')[0]
                if game_type not in game_type_trials:
                    game_type_trials[game_type] = []
                game_type_trials[game_type].append(trial)
            
            # Select approximately equal number of trials from each game type
            selected_trials = []
            game_types = list(game_type_trials.keys())
            while len(selected_trials) < num_trials and game_types:
                for game_type in list(game_types):  # Use list() to avoid modification during iteration
                    if game_type_trials[game_type]:
                        trial = random.choice(game_type_trials[game_type])
                        game_type_trials[game_type].remove(trial)
                        selected_trials.append(trial)
                        if len(selected_trials) >= num_trials:
                            break
                    else:
                        game_types.remove(game_type)
            
            return selected_trials
    
    def prepare_one_shot_examples(self, test_trial: Path, strategy: str) -> List[Dict]:
        """
        根据策略准备1-shot示例，包含中间帧
        
        Args:
            test_trial: 要测试的试验
            strategy: 示例选择策略 ('success_example' 或 'failure_example')
        
        Returns:
            示例列表(包含一个示例)，示例包含初始帧、中间帧1、中间帧2和最终帧
        """
        # 获取测试试验的游戏类型
        test_game_type = test_trial.name.split('_')[0]
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验和已经使用的示例
        all_trials = [t for t in all_trials if t != test_trial and str(t) not in self.used_examples]
        
        if strategy == "success_example":
            # 先尝试找同类型的成功案例
            candidates = [t for t in all_trials 
                         if t.name.split('_')[0] == test_game_type 
                         and t.name.endswith("True")]
            
            # 如果找不到同类型的成功案例，则从所有成功案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("True")]
            
            if not candidates:
                print(f"警告: 找不到成功案例，跳过此策略")
                return []
        
        elif strategy == "failure_example":
            # 先尝试找同类型的失败案例
            candidates = [t for t in all_trials 
                         if t.name.split('_')[0] == test_game_type 
                         and t.name.endswith("False")]
            
            # 如果找不到同类型的失败案例，则从所有失败案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("False")]
            
            if not candidates:
                print(f"警告: 找不到失败案例，跳过此策略")
                return []
        
        else:
            print(f"错误: 未知的策略 {strategy}")
            return []
        
        # 随机选择一个候选案例
        random.shuffle(candidates)
        
        # 尝试多个候选，直到找到有效的
        for trial in candidates:
            try:
                # 获取所有帧并按序号排序
                all_frames = sorted(list(trial.glob("frame_*.png")), 
                                  key=lambda p: int(p.stem.split('_')[1]))
                
                # 确保有足够的帧
                if len(all_frames) < 4:
                    print(f"警告: {trial} 中帧数量不足(<4)，跳过")
                    continue
                
                # 获取初始帧和最终帧
                first_frame = all_frames[0]
                last_frame = all_frames[-1]
                
                # 计算中间帧索引（选择整个序列的1/3和2/3位置）
                frame_count = len(all_frames)
                middle_idx1 = frame_count // 3
                middle_idx2 = (frame_count * 2) // 3
                middle_frame1 = all_frames[middle_idx1]
                middle_frame2 = all_frames[middle_idx2]
                
                game_type = trial.name.split('_')[0]
                is_success = trial.name.endswith("True")
                
                # 将选中的示例添加到已使用集合中
                self.used_examples.add(str(trial))
                
                return [{
                    "initial_frame": str(first_frame),
                    "middle_frame1": str(middle_frame1),
                    "middle_frame2": str(middle_frame2),
                    "final_frame": str(last_frame),
                    "is_success": is_success,
                    "game_type": game_type,
                    "description": f"In this {self.game_type_desc.get(game_type, game_type)}, the red ball {'successfully reached' if is_success else 'failed to reach'} the green target."
                }]
            except Exception as e:
                print(f"警告: 处理 {trial} 时出错: {e}，跳过")
                continue
        
        # 如果所有候选都无效，返回空列表
        return []
    
    def build_prompt_with_examples(self, test_trial: Path, examples: List[Dict]) -> Tuple[List, bool, str]:
        """
        Build prompt with specified examples, using scenario-specific prompts from prompt1 folder
        
        Args:
            test_trial: Trial to test
            examples: List of examples to use
            
        Returns:
            Prompt message list, true result and game type
        """
        # Try to get the first frame of the test trial
        try:
            first_frame = next(test_trial.glob("frame_0000.png"))
        except StopIteration:
            print(f"Warning: Could not find first frame image in {test_trial}, skipping this trial")
            return None, None, None
        
        # Get true result
        true_result = test_trial.name.endswith("True")
        # 提取游戏类型，移除可能的A或B后缀
        game_type = test_trial.name.split('_')[0]
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        messages = []
        
        # System prompt - use scenario-specific prompt if available
        system_content = "You are an AI assistant with strong physical intuition. You need to predict whether the red ball will reach the green target area based on the physical scene image. Start your answer with a clear YES or NO before providing explanation."
        
        if game_type in self.scenario_prompts:
            # Use the full scenario-specific prompt from prompt1 folder
            system_content = self.scenario_prompts[game_type]
        elif base_game_type in self.scenario_prompts:
            # 如果找不到带后缀的游戏类型，尝试使用基本游戏类型
            system_content = self.scenario_prompts[base_game_type]
        
        system_message = {
            "role": "system", 
            "content": system_content
        }
        messages.append(system_message)
        
        # Add examples
        for i, example in enumerate(examples):
            base64_initial = encode_image_to_base64(example['initial_frame'])
            base64_final = encode_image_to_base64(example['final_frame'])
            
            # 构建示例内容
            example_content = []
            
            # 添加初始帧
            example_content.append({"type": "text", "text": f"Initial state of Example {i+1}:"})
            example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_initial}"}})
            
            # 添加中间帧（如果有）
            if 'middle_frame1' in example and 'middle_frame2' in example:
                base64_middle1 = encode_image_to_base64(example['middle_frame1'])
                base64_middle2 = encode_image_to_base64(example['middle_frame2'])
                
                example_content.append({"type": "text", "text": f"Process frame 1 of Example {i+1}:"})
                example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_middle1}"}})
                example_content.append({"type": "text", "text": f"Process frame 2 of Example {i+1}:"})
                example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_middle2}"}})
            
            # 添加最终帧
            example_content.append({"type": "text", "text": f"Final state of Example {i+1}:"})
            example_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_final}"}})
            example_content.append({"type": "text", "text": example["description"]})
            
            messages.append({"role": "user", "content": example_content})
            messages.append({"role": "assistant", "content": f"{'YES' if example['is_success'] else 'NO'}, in this {self.game_type_desc.get(example['game_type'], example['game_type'])}, the red ball {'successfully reached' if example['is_success'] else 'failed to reach'} the green target area."})
        
        # Add test question
        if not examples:  # For 0-shot
            # Extract the question part from the scenario prompt if available
            question_text = "Based on the initial scene image, will the red ball eventually reach the green target area? Start your answer with a clear YES or NO, then explain your reasoning process in detail."
        else:  # For few-shot
            question_text = "Based on the previous examples and your physical knowledge, will the red ball eventually reach the green target area? Start your answer with a clear YES or NO, then explain your reasoning process in detail."
        
        base64_first = encode_image_to_base64(str(first_frame))
        question_content = [
            {"type": "text", "text": question_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_first}"}}
        ]
        
        messages.append({"role": "user", "content": question_content})
        
        return messages, true_result, game_type
    
    def run_controlled_shot_experiment(self, num_trials=20, temperature=0.0):
        """
        Run a strictly controlled few-shot experiment with 11 conditions:
        
        1. **0-shot baseline**: No examples
        2. **1-shot conditions**:
           - success_1shot: 1 successful example
           - failure_1shot: 1 failed example
        3. **2-shot conditions**:
           - success_success_2shot: 2 successful examples
           - success_failure_2shot: 1 successful + 1 failed example  
           - failure_success_2shot: 1 failed + 1 successful example
           - failure_failure_2shot: 2 failed examples
        4. **3-shot conditions** (based on best 2-shot strategies):
           - best2shot_success_3shot: Best 2-shot strategy + 1 successful example
           - best2shot_failure_3shot: Best 2-shot strategy + 1 failed example
        
        All conditions use the same test set for fair comparison.
        
        Args:
            num_trials: Number of trials to test for each condition
            temperature: Temperature parameter for the model (0.0-1.0)
        """
        # Reset results
        self.results = []
        
        # Generate timestamp for this experiment run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save experiment parameters for reproducibility
        experiment_params = {
            "experiment_type": "controlled_experiment",
            "num_trials": num_trials,
            "temperature": temperature,
            "model_name": self.model_name,
            "timestamp": timestamp,
            "balance_success_failure": True  # Force balanced selection for fairness
        }
        
        with open(self.result_dir / "experiment_params.json", 'w', encoding='utf-8') as f:
            json.dump(experiment_params, f, ensure_ascii=False, indent=2)
        
        # Select diverse trial samples - ensure balance between success and failure
        print(f"Selecting {num_trials} diverse trials with balanced success/failure cases...")
        test_trials = self.select_diverse_trials(num_trials, balance_success=True)
        
        # Log trial selection for verification
        trial_info = [{"name": t.name, 
                       "game_type": t.name.split('_')[0], 
                       "is_success": t.name.endswith("True")} for t in test_trials]
        
        with open(self.result_dir / "selected_trials.json", 'w', encoding='utf-8') as f:
            json.dump(trial_info, f, ensure_ascii=False, indent=2)
        
        # Count success and failure cases
        success_count = sum(1 for t in test_trials if t.name.endswith("True"))
        failure_count = sum(1 for t in test_trials if t.name.endswith("False"))
        print(f"Selected trials: {success_count} success cases, {failure_count} failure cases")
        
        # Print game type distribution
        game_type_counts = {}
        for trial in test_trials:
            game_type = trial.name.split('_')[0]
            game_type_counts[game_type] = game_type_counts.get(game_type, 0) + 1
        
        print("Game type distribution:")
        for game_type, count in game_type_counts.items():
            print(f"  {game_type}: {count}")
            
        # Define phase 1 conditions (0-shot, 1-shot, 2-shot)
        base_conditions = [
            "0shot",
            "success_1shot",
            "failure_1shot",
            "success_success_2shot",
            "success_failure_2shot", 
            "failure_success_2shot",
            "failure_failure_2shot"
        ]
        
        # Phase 1: Run basic 0-shot, 1-shot, and 2-shot conditions
        for i, trial in enumerate(test_trials):
            print(f"\n===== Testing trial {i+1}/{len(test_trials)}: {trial.name} =====")
            
            # Get trial type information
            game_type = trial.name.split('_')[0]
            
            # Prepare 1-shot examples for this trial
            success_example = self.prepare_one_shot_examples(trial, "success_example")
            failure_example = self.prepare_one_shot_examples(trial, "failure_example")
            
            # If unable to prepare basic examples, skip this trial
            if not success_example or not failure_example:
                print(f"  Warning: Unable to prepare basic examples for this trial, skipping")
                    continue
                
            # Prepare 2-shot example combinations
            success_success_examples = self.prepare_two_shot_examples(
                trial, success_example, "success_example")
            success_failure_examples = self.prepare_two_shot_examples(
                trial, success_example, "failure_example")
            failure_success_examples = self.prepare_two_shot_examples(
                trial, failure_example, "success_example")
            failure_failure_examples = self.prepare_two_shot_examples(
                trial, failure_example, "failure_example")
            
            # Run each base condition
            for condition in base_conditions:
                print(f"  Condition: {condition}")
                
                # Choose examples based on condition
                if condition == "0shot":
                    examples = []
                elif condition == "success_1shot":
                    examples = success_example
                elif condition == "failure_1shot":
                    examples = failure_example
                elif condition == "success_success_2shot":
                    examples = success_success_examples
                elif condition == "success_failure_2shot":
                    examples = success_failure_examples
                elif condition == "failure_success_2shot":
                    examples = failure_success_examples
                elif condition == "failure_failure_2shot":
                    examples = failure_failure_examples
                
                # Build prompt
                messages, true_result, _ = self.build_prompt_with_examples(trial, examples)
                if messages is None:
                    print(f"    Warning: Unable to build prompt, skipping this condition")
                    continue
                
                # Call model
                response = self.call_model(messages, temperature)
                if not response:
                    print(f"    Error: Model call failed")
                    continue
                
                # Parse prediction
                prediction = response[0]
                explanation = response[1]
                
                # Record result
                result = {
                    "trial": trial.name,
                    "game_type": game_type,
                    "condition": condition,
                    "shot_count": len(examples),
                    "true_result": true_result,
                    "prediction": prediction,
                    "correct": prediction == true_result,
                    "response": explanation
                }
                
                self.results.append(result)
                print(f"    Result: {'CORRECT' if prediction == true_result else 'INCORRECT'} (Predicted: {'Success' if prediction else 'Failure'}, Actual: {'Success' if true_result else 'Failure'})")
                
                # Save intermediate results
                intermediate_filename = f"controlled_experiment_intermediate_temp{temperature}_trials{len(self.results)}.json"
                self.save_results(intermediate_filename)
                
                # Brief pause to avoid API limits
                time.sleep(1)
        
        # Save phase 1 results
        phase1_filename = f"controlled_experiment_phase1_temp{temperature}_trials{len(self.results)}.json"
        self.save_results(phase1_filename)
        
        # Analyze 2-shot results to determine best and second-best strategies
        df = pd.DataFrame(self.results)
        # 检查是否包含 'condition' 列
        if 'condition' not in df.columns:
            raise ValueError("The 'condition' column is missing from the results.")
        two_shot_conditions = [c for c in base_conditions if c.endswith('2shot')]
        
        two_shot_df = df[df['condition'].isin(two_shot_conditions)]
        two_shot_accuracy = two_shot_df.groupby('condition')['correct'].mean()
        
        if len(two_shot_accuracy) >= 2:
            # Sort to find best and second-best 2-shot strategies
            sorted_conditions = two_shot_accuracy.sort_values(ascending=False).index.tolist()
            best_2shot = sorted_conditions[0]
            
            print(f"\n===== 2-Shot Strategy Analysis =====")
            print(f"Best 2-shot strategy: {best_2shot}, accuracy: {two_shot_accuracy[best_2shot]:.4f}")
            
            # Phase 2: Build 3-shot experiments based on best 2-shot strategies
            three_shot_conditions = [
                f"best2shot_success_3shot",
                f"best2shot_failure_3shot"
            ]
            
            # Map condition names to actual data
            condition_map = {
                "best2shot_success_3shot": {"base": best_2shot, "add": "success_example"},
                "best2shot_failure_3shot": {"base": best_2shot, "add": "failure_example"}
            }
            
            # Phase 2: Run 3-shot experiments
            print("\n===== Phase 2: 3-Shot Experiments =====")
            for i, trial in enumerate(test_trials):
                print(f"\n===== Testing trial {i+1}/{len(test_trials)}: {trial.name} =====")
                
                # Get trial type information
                game_type = trial.name.split('_')[0]
                
                # Skip if we already detected issues with this trial in Phase 1
                if trial.name not in df['trial'].values:
                    print(f"  Warning: Trial {trial.name} was skipped in Phase 1, skipping in Phase 2 as well")
                    continue
                
                # Get basic examples again
                success_example = self.prepare_one_shot_examples(trial, "success_example")
                failure_example = self.prepare_one_shot_examples(trial, "failure_example")
                
                # Get 2-shot examples again
                two_shot_examples = {
                    "success_success_2shot": self.prepare_two_shot_examples(trial, success_example, "success_example"),
                    "success_failure_2shot": self.prepare_two_shot_examples(trial, success_example, "failure_example"),
                    "failure_success_2shot": self.prepare_two_shot_examples(trial, failure_example, "success_example"),
                    "failure_failure_2shot": self.prepare_two_shot_examples(trial, failure_example, "failure_example")
                }
                
                # Run each 3-shot condition
                for condition in three_shot_conditions:
                    print(f"  Condition: {condition}")
                    
                    # Get the base 2-shot condition and additional example type
                    base_condition = condition_map[condition]["base"]
                    add_example_type = condition_map[condition]["add"]
                    
                    # Get the base 2-shot examples
                    base_examples = two_shot_examples[base_condition]
                    
                    # Prepare the 3-shot examples
                    third_example = self.prepare_one_shot_examples(trial, add_example_type)
                    if not third_example:
                        print(f"    Warning: Unable to prepare additional example, skipping this condition")
                        continue
                    
                    # Combine examples
                    three_shot_examples = self.prepare_three_shot_examples(trial, base_examples, add_example_type)
                    
                    # Build prompt
                    messages, true_result, _ = self.build_prompt_with_examples(trial, three_shot_examples)
                    if messages is None:
                        print(f"    Warning: Unable to build prompt, skipping this condition")
                        continue
                    
                    # Call model
                    response = self.call_model(messages, temperature)
                    if not response:
                        print(f"    Error: Model call failed")
                        continue
                    
                    # Parse prediction
                    prediction = response[0]
                    explanation = response[1]
                    
                    # Record result
                    result = {
                        "trial": trial.name,
                        "game_type": game_type,
                        "condition": condition,
                        "shot_count": 3,
                        "base_condition": base_condition,
                        "true_result": true_result,
                        "prediction": prediction,
                        "correct": prediction == true_result,
                        "response": explanation
                    }
                    
                    self.results.append(result)
                    print(f"    Result: {'CORRECT' if prediction == true_result else 'INCORRECT'} (Predicted: {'Success' if prediction else 'Failure'}, Actual: {'Success' if true_result else 'Failure'})")
                    
                    # Save intermediate results
                    intermediate_filename = f"controlled_experiment_intermediate_temp{temperature}_trials{len(self.results)}.json"
                    self.save_results(intermediate_filename)
                    
                    # Brief pause to avoid API limits
                    time.sleep(1)
        
        # Save final results
        final_filename = f"controlled_experiment_final_temp{temperature}_trials{len(self.results)}.json"
        self.save_results(final_filename)
        
        # Analyze results
        self.analyze_controlled_experiment()
        
        return self.results
    
    def prepare_two_shot_examples(self, test_trial: Path, first_shot_examples: List[Dict], 
                               second_shot_strategy: str) -> List[Dict]:
        """
        在1-shot示例的基础上，准备2-shot示例
        
        Args:
            test_trial: 要测试的试验
            first_shot_examples: 第一个shot的示例列表
            second_shot_strategy: 第二个示例的选择策略 ('success_example' 或 'failure_example')
            
        Returns:
            包含两个示例的列表，每个示例包含初始帧、中间帧1、中间帧2和最终帧
        """
        # 复制第一个shot的示例
        examples = first_shot_examples.copy()
        
        # 获取测试试验的游戏类型
        test_game_type = test_trial.name.split('_')[0]
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验和已经使用的示例
        used_trials = [Path(ex['initial_frame']).parent for ex in examples]
        all_trials = [t for t in all_trials if t != test_trial and t not in used_trials and str(t) not in self.used_examples]
        
        if second_shot_strategy == "success_example":
            # 先尝试找同类型的成功案例
            candidates = [t for t in all_trials 
                         if t.name.split('_')[0] == test_game_type 
                         and t.name.endswith("True")]
            
            # 如果找不到同类型的成功案例，则从所有成功案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("True")]
        
        elif second_shot_strategy == "failure_example":
            # 先尝试找同类型的失败案例
                candidates = [t for t in all_trials 
                              if t.name.split('_')[0] == test_game_type 
                              and t.name.endswith("False")]
            
            # 如果找不到同类型的失败案例，则从所有失败案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("False")]
        
        # 如果没有合适的候选，返回原始示例
        if not candidates:
            print(f"  警告: 找不到适合{second_shot_strategy}的第二个示例，使用原始1-shot示例")
            return examples
        
        # 随机选择一个候选案例
            random.shuffle(candidates)
        
        # 尝试多个候选，直到找到有效的
        for trial in candidates:
            try:
                # 获取所有帧并按序号排序
                all_frames = sorted(list(trial.glob("frame_*.png")), 
                                  key=lambda p: int(p.stem.split('_')[1]))
                
                # 确保有足够的帧
                if len(all_frames) < 4:
                    print(f"  警告: {trial} 中帧数量不足(<4)，跳过")
                    continue
                
                # 获取初始帧和最终帧
                first_frame = all_frames[0]
                last_frame = all_frames[-1]
                
                # 计算中间帧索引（选择整个序列的1/3和2/3位置）
                frame_count = len(all_frames)
                middle_idx1 = frame_count // 3
                middle_idx2 = (frame_count * 2) // 3
                middle_frame1 = all_frames[middle_idx1]
                middle_frame2 = all_frames[middle_idx2]
                
                game_type = trial.name.split('_')[0]
                is_success = trial.name.endswith("True")
                
                # 将选中的示例添加到已使用集合中
                self.used_examples.add(str(trial))
                
                second_example = {
                    "initial_frame": str(first_frame),
                    "middle_frame1": str(middle_frame1),
                    "middle_frame2": str(middle_frame2),
                    "final_frame": str(last_frame),
                    "is_success": is_success,
                    "game_type": game_type,
                    "description": f"In this {self.game_type_desc.get(game_type, game_type)}, the red ball {'successfully reached' if is_success else 'failed to reach'} the green target."
                }
                
                # 添加第二个示例
                examples.append(second_example)
                return examples
                
            except Exception as e:
                print(f"  警告: 处理 {trial} 时出错: {e}，尝试下一个候选")
                continue
        
        # 如果所有候选都无效，返回原始示例
        print(f"  警告: 所有候选都无效，使用原始1-shot示例")
        return examples
    
    def prepare_three_shot_examples(self, test_trial: Path, two_shot_examples: List[Dict], 
                               third_shot_strategy: str) -> List[Dict]:
        """
        在2-shot示例的基础上，准备3-shot示例
        
        Args:
            test_trial: 要测试的试验
            two_shot_examples: 前两个shot的示例列表
            third_shot_strategy: 第三个示例的选择策略 ('success_example' 或 'failure_example')
            
        Returns:
            包含三个示例的列表，每个示例包含初始帧、中间帧1、中间帧2和最终帧
        """
        # 复制前两个shot的示例
        examples = two_shot_examples.copy()
        
        # 获取测试试验的游戏类型
        test_game_type = test_trial.name.split('_')[0]
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验和已经使用的示例
        used_trials = [Path(ex['initial_frame']).parent for ex in examples]
        all_trials = [t for t in all_trials if t != test_trial and t not in used_trials and str(t) not in self.used_examples]
        
        if third_shot_strategy == "success_example":
            # 先尝试找同类型的成功案例
            candidates = [t for t in all_trials 
                                     if t.name.split('_')[0] == test_game_type 
                                     and t.name.endswith("True")]
                
            # 如果找不到同类型的成功案例，则从所有成功案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("True")]
        
        elif third_shot_strategy == "failure_example":
            # 先尝试找同类型的失败案例
            candidates = [t for t in all_trials 
                                     if t.name.split('_')[0] == test_game_type 
                                     and t.name.endswith("False")]
            
            # 如果找不到同类型的失败案例，则从所有失败案例中选择
            if not candidates:
                candidates = [t for t in all_trials if t.name.endswith("False")]
        
        # 如果没有合适的候选，返回原始示例
        if not candidates:
            print(f"  警告: 找不到适合{third_shot_strategy}的第三个示例，使用原始2-shot示例")
            return examples
        
        # 随机选择一个候选案例
        random.shuffle(candidates)
        
        # 尝试多个候选，直到找到有效的
        for trial in candidates:
            try:
                # 获取所有帧并按序号排序
                all_frames = sorted(list(trial.glob("frame_*.png")), 
                                  key=lambda p: int(p.stem.split('_')[1]))
                
                # 确保有足够的帧
                if len(all_frames) < 4:
                    print(f"  警告: {trial} 中帧数量不足(<4)，跳过")
                    continue
                
                # 获取初始帧和最终帧
                first_frame = all_frames[0]
                last_frame = all_frames[-1]
                
                # 计算中间帧索引（选择整个序列的1/3和2/3位置）
                frame_count = len(all_frames)
                middle_idx1 = frame_count // 3
                middle_idx2 = (frame_count * 2) // 3
                middle_frame1 = all_frames[middle_idx1]
                middle_frame2 = all_frames[middle_idx2]
                
                game_type = trial.name.split('_')[0]
                is_success = trial.name.endswith("True")
                
                # 将选中的示例添加到已使用集合中
                self.used_examples.add(str(trial))
                
                third_example = {
                    "initial_frame": str(first_frame),
                    "middle_frame1": str(middle_frame1),
                    "middle_frame2": str(middle_frame2),
                    "final_frame": str(last_frame),
                    "is_success": is_success,
                    "game_type": game_type,
                    "description": f"In this {self.game_type_desc.get(game_type, game_type)}, the red ball {'successfully reached' if is_success else 'failed to reach'} the green target."
                }
                
                # 添加第三个示例
                examples.append(third_example)
                return examples
                
            except Exception as e:
                print(f"  警告: 处理 {trial} 时出错: {e}，尝试下一个候选")
                continue
        
        # 如果所有候选都无效，返回原始示例
        print(f"  警告: 所有候选都无效，使用原始2-shot示例")
        return examples
    
    def analyze_controlled_experiment(self):
        """
        Analyze the results of the controlled experiment.
        
        This function:
        1. Calculates accuracy for each condition
        2. Compares performance across shot counts (0, 1, 2, 3)
        3. Analyzes the effect of example patterns (success vs. failure)
        4. Generates visualizations and saves statistics to files
        """
        if not self.results:
            print("No results to analyze")
            return
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        print("\n===== Controlled Experiment Analysis =====")
        print(f"Total samples analyzed: {len(df)}")
        
        # Verify all 11 conditions are present
        all_conditions = [
            "0shot", 
            "success_1shot", "failure_1shot",
            "success_success_2shot", "success_failure_2shot", 
            "failure_success_2shot", "failure_failure_2shot",
            "best2shot_success_3shot", "best2shot_failure_3shot"
        ]
        
        found_conditions = df['condition'].unique()
        print(f"Conditions found in results: {len(found_conditions)}")
        for condition in all_conditions:
            if condition in found_conditions:
                print(f"  ✓ {condition}")
            else:
                print(f"  ✗ {condition} (missing)")
        
        # Calculate overall accuracy
        overall_accuracy = df['correct'].mean()
        print(f"\nOverall accuracy: {overall_accuracy:.4f}")
        
        # 1. Analyze by condition
        condition_accuracy = df.groupby('condition')['correct'].agg(['mean', 'count']).reset_index()
        condition_accuracy.columns = ['Condition', 'Accuracy', 'Count']
        condition_accuracy = condition_accuracy.sort_values('Accuracy', ascending=False)
        
        print("\nAccuracy by condition (sorted by accuracy):")
        for i, row in condition_accuracy.iterrows():
            print(f"  {row['Condition']}: {row['Accuracy']:.4f} (n={int(row['Count'])})")
        
        # 2. Analyze by shot count
        def get_shot_count(condition):
            if condition == "0shot":
                return 0
            elif condition.endswith("1shot"):
                return 1
            elif condition.endswith("2shot"):
                return 2
            elif condition.endswith("3shot"):
                return 3
            else:
                return -1
        
        df['shot_count'] = df['condition'].apply(get_shot_count)
        shot_accuracy = df.groupby('shot_count')['correct'].agg(['mean', 'count']).reset_index()
        shot_accuracy.columns = ['Shot Count', 'Accuracy', 'Count']
        shot_accuracy = shot_accuracy.sort_values('Shot Count')
        
        print("\nAccuracy by shot count:")
        for i, row in shot_accuracy.iterrows():
            print(f"  {int(row['Shot Count'])}-shot: {row['Accuracy']:.4f} (n={int(row['Count'])})")
        
        # 3. Analyze by example pattern
        def get_example_pattern(condition):
            if condition == "0shot":
                return "No Examples"
            elif condition == "success_1shot":
                return "Success Only"
            elif condition == "failure_1shot":
                return "Failure Only"
            elif condition == "success_success_2shot":
                return "All Success"
            elif condition == "failure_failure_2shot":
                return "All Failure"
            elif condition in ["success_failure_2shot", "failure_success_2shot"]:
                return "Mixed Success/Failure"
            elif "best2shot" in condition:
                # For 3-shot conditions, check if they end with "success" or "failure"
                if condition.endswith("success_3shot"):
                    return "Best + Success"
                elif condition.endswith("failure_3shot"):
                    return "Best + Failure"
            return "Other"
        
        df['example_pattern'] = df['condition'].apply(get_example_pattern)
        pattern_accuracy = df.groupby(['shot_count', 'example_pattern'])['correct'].agg(['mean', 'count']).reset_index()
        pattern_accuracy.columns = ['Shot Count', 'Example Pattern', 'Accuracy', 'Count']
        pattern_accuracy = pattern_accuracy.sort_values(['Shot Count', 'Accuracy'], ascending=[True, False])
        
        print("\nAccuracy by example pattern:")
        for sc in sorted(pattern_accuracy['Shot Count'].unique()):
            print(f"  {int(sc)}-shot patterns:")
            subset = pattern_accuracy[pattern_accuracy['Shot Count'] == sc]
            for i, row in subset.iterrows():
                print(f"    {row['Example Pattern']}: {row['Accuracy']:.4f} (n={int(row['Count'])})")
        
        # 4. Analyze by game type
        game_type_accuracy = df.groupby(['game_type', 'condition'])['correct'].mean().unstack().reset_index()
        print("\nAccuracy by game type and condition:")
        print(game_type_accuracy)
        
        # Save numeric results to text file
        with open(self.result_dir / 'controlled_experiment_analysis.txt', 'w') as f:
            f.write("===== CONTROLLED EXPERIMENT ANALYSIS =====\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Overall accuracy: {overall_accuracy:.4f}\n\n")
            
            f.write("Accuracy by condition:\n")
            for i, row in condition_accuracy.iterrows():
                f.write(f"  {row['Condition']}: {row['Accuracy']:.4f} (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy by shot count:\n")
            for i, row in shot_accuracy.iterrows():
                f.write(f"  {int(row['Shot Count'])}-shot: {row['Accuracy']:.4f} (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy by example pattern:\n")
            for sc in sorted(pattern_accuracy['Shot Count'].unique()):
                f.write(f"  {int(sc)}-shot patterns:\n")
                subset = pattern_accuracy[pattern_accuracy['Shot Count'] == sc]
                for i, row in subset.iterrows():
                    f.write(f"    {row['Example Pattern']}: {row['Accuracy']:.4f} (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy by game type and condition:\n")
            f.write(game_type_accuracy.to_string())
        
        # Create visualizations
        plt.figure(figsize=(12, 6))
        
        # Sort conditions in a logical order for visualization
        ordered_conditions = []
        for prefix in ["0shot", "success_1shot", "failure_1shot", 
                     "success_success_2shot", "success_failure_2shot", 
                     "failure_success_2shot", "failure_failure_2shot"]:
            if prefix in found_conditions:
                ordered_conditions.append(prefix)
        
        # Add 3-shot conditions at the end
        for condition in found_conditions:
            if "3shot" in condition and condition not in ordered_conditions:
                ordered_conditions.append(condition)
        
        # Plot condition accuracy
        condition_df = condition_accuracy.copy()
        condition_df['Condition'] = pd.Categorical(
            condition_df['Condition'], 
            categories=ordered_conditions,
            ordered=True
        )
        condition_df = condition_df.sort_values('Condition')
        
        ax = plt.subplot(111)
        bars = plt.bar(condition_df['Condition'], condition_df['Accuracy'], color='skyblue')
        
        # Color-code bars by shot count
        colors = ['#FFC107', '#4CAF50', '#2196F3', '#9C27B0']  # 0-shot, 1-shot, 2-shot, 3-shot
        for i, bar in enumerate(bars):
            condition = condition_df.iloc[i]['Condition']
            shot_count = get_shot_count(condition)
            bar.set_color(colors[shot_count])
        
        plt.title('Accuracy by Experimental Condition')
        plt.xlabel('Condition')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)  # Chance level
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label='0-shot'),
            Patch(facecolor=colors[1], label='1-shot'),
            Patch(facecolor=colors[2], label='2-shot'),
            Patch(facecolor=colors[3], label='3-shot')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(self.result_dir / 'accuracy_by_condition.png', dpi=300, bbox_inches='tight')
        
        # Plot accuracy by shot count
        plt.figure(figsize=(8, 6))
        plt.bar(shot_accuracy['Shot Count'].astype(str) + '-shot', shot_accuracy['Accuracy'], color=['#FFC107', '#4CAF50', '#2196F3', '#9C27B0'])
        plt.title('Accuracy by Shot Count')
        plt.xlabel('Shot Count')
        plt.ylabel('Accuracy')
                plt.ylim(0, 1)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)  # Chance level
        
        # Add value labels
        for i, v in enumerate(shot_accuracy['Accuracy']):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.savefig(self.result_dir / 'accuracy_by_shot_count.png', dpi=300, bbox_inches='tight')
        
        # Plot success vs. failure vs. mixed examples effect
        plt.figure(figsize=(10, 8))
        
        # Filter to just show 1-shot and 2-shot for pattern comparison
        pattern_subset = pattern_accuracy[pattern_accuracy['Shot Count'].isin([1, 2])]
        pattern_subset = pattern_subset.sort_values(['Shot Count', 'Accuracy'], ascending=[True, False])
        
        # Create indexed bars
        indices = np.arange(len(pattern_subset))
        bars = plt.bar(indices, pattern_subset['Accuracy'])
        
        # Color by pattern type
        pattern_colors = {
            'Success Only': '#4CAF50',
            'Failure Only': '#F44336',
            'All Success': '#4CAF50',
            'All Failure': '#F44336',
            'Mixed Success/Failure': '#2196F3',
        }
        
        for i, bar in enumerate(bars):
            pattern = pattern_subset.iloc[i]['Example Pattern']
            if pattern in pattern_colors:
                bar.set_color(pattern_colors[pattern])
        
        plt.title('Accuracy by Example Pattern Type')
        plt.xlabel('Example Pattern')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)  # Chance level
        
        # Customize x-axis
        plt.xticks(indices, [f"{int(row['Shot Count'])}-shot: {row['Example Pattern']}" 
                             for _, row in pattern_subset.iterrows()], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'accuracy_by_pattern.png', dpi=300, bbox_inches='tight')
        
        print("\nAnalysis complete. Results saved to the output directory.")
        return condition_accuracy

def main():
    """
    Main entry point for the physical intuition evaluator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--model_name", type=str, default="qwen2.5-omni-7b", help="模型名称")
    parser.add_argument("--api_key", type=str, default=None, help="API密钥")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API基础URL")
    parser.add_argument("--output_dir", type=str, default="results1", help="输出目录")
    parser.add_argument("--prompt_dir", type=str, default="prompt1", help="提示目录")
    parser.add_argument("--result_folder", type=str, default=None, help="结果文件夹名称")
    parser.add_argument("--num_trials", type=int, default=10, help="试验次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建评估器
    evaluator = PhysicalIntuitionEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        result_folder=args.result_folder
    )
    
    # 直接运行控制实验，移除eval_type判断
    evaluator.run_controlled_shot_experiment(num_trials=args.num_trials, temperature=args.temperature)

if __name__ == "__main__":
    main()