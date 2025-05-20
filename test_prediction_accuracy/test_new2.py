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
                 output_dir: str = "results1",
                 prompt_dir: str = "prompt1",
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
            api_key="ollama",  # required but unused for Ollama
            base_url="http://localhost:11434/v1",
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
            "Falling": "falling object scenario",
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
        
        # No need to track used examples as we can reuse examples across trials
        # self.used_examples = set()
    
    def _load_scenario_prompts(self):
        """Load scenario-specific prompts from the prompt directory"""
        # 确保prompt_dir是绝对路径
        prompt_dir = Path(self.prompt_dir)
        if not prompt_dir.is_absolute():
            prompt_dir = Path.cwd() / prompt_dir
            
        print(f"Loading prompts from: {prompt_dir}")
        
        # 检查文件夹是否存在
        if not prompt_dir.exists() or not prompt_dir.is_dir():
            print(f"Warning: Prompt directory not found: {prompt_dir}")
            return
            
        # 列出目录中的所有文件，用于调试
        prompt_files = list(prompt_dir.glob("*.txt"))
        print(f"Found prompt files: {[f.name for f in prompt_files]}")
        
        for game_type in self.game_type_desc.keys():
            # 移除可能的A或B后缀，只保留基本游戏类型名称
            base_game_type = game_type.split('A')[0].split('B')[0]
            prompt_file = prompt_dir / f"{base_game_type}.txt"
            
            print(f"Looking for prompt file: {prompt_file}")
            
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r') as f:
                        self.scenario_prompts[game_type] = f.read().strip()
                        print(f"Successfully loaded prompt for {game_type}")
                except Exception as e:
                    print(f"Warning: Could not load prompt for {game_type}: {e}")
            else:
                print(f"Warning: No prompt file found for {game_type} at {prompt_file}")
    
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
        Call model and get prediction result
        
        Args:
            messages: List of messages
            temperature: Temperature parameter
            
        Returns:
            prediction: Prediction result (True for success, False for failure)
            explanation: Explanation
            response_time: Time taken to get a response from the model (in seconds)
        """
        try:
            # Check if messages is a valid list
            if not messages or not isinstance(messages, list):
                print(f"Error: Invalid message format: {messages}")
                return None, None, None
            
            # Start timing the API call
            start_time = time.time()
                
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract prediction and explanation
            response_text = response.choices[0].message.content
            prediction = self.parse_prediction(response_text)
            explanation = response_text
            
            return prediction, explanation, response_time
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return None, None, None
            
    def parse_prediction(self, response_text):
        """
        Parse prediction result from model response
        
        Args:
            response_text: Model's response text
            
        Returns:
            prediction: Prediction result (True for success, False for failure)
        """
        # Convert text to lowercase and remove special characters for more precise matching
        text = response_text.lower().strip()
        
        # Success pattern matching
        success_patterns = [
            r'\byes\b',
            r'\bsuccess\b', 
            r'\bwill\s+succeed\b',
            r'\bwould\s+succeed\b', 
            r'\bis\s+successful\b',
            r'\bwill\s+be\s+successful\b',
            r'\bwill\s+work\b',
            r'\bworks\b',
            r'\bwill\s+accomplish\b',
            r'\baccomplishes\b',
            r'\bachieved\b',
            r'\bwill\s+achieve\b',
            r'\breaches\b',
            r'\bwill\s+reach\b',
            r'\bfulfills\b',
            r'\bwill\s+fulfill\b',
            r'\bcompletes\b',
            r'\bwill\s+complete\b'
        ]
        
        # Failure pattern matching
        failure_patterns = [
            r'\bno\b',
            r'\bfailure\b',
            r'\bfail\b',
            r'\bwill\s+fail\b',
            r'\bwon\'?t\s+succeed\b',
            r'\bnot\s+successful\b',
            r'\bnot\s+work\b',
            r'\bwon\'?t\s+work\b',
            r'\bis\s+unsuccessful\b',
            r'\bwill\s+be\s+unsuccessful\b',
            r'\bwon\'?t\s+accomplish\b',
            r'\bnot\s+accomplish\b',
            r'\bwon\'?t\s+achieve\b',
            r'\bnot\s+achieve\b',
            r'\bwon\'?t\s+reach\b',
            r'\bnot\s+reach\b',
            r'\bwon\'?t\s+fulfill\b',
            r'\bnot\s+fulfill\b',
            r'\bwon\'?t\s+complete\b',
            r'\bnot\s+complete\b'
        ]
        
        # Check for success and failure patterns separately
        success_match = False
        failure_match = False
        
        # Check if matching any success pattern
        for pattern in success_patterns:
            if re.search(pattern, text):
                success_match = True
                break
            
        # Check if matching any failure pattern
        for pattern in failure_patterns:
            if re.search(pattern, text):
                failure_match = True
                break
        
        # If success pattern found and no failure pattern, it's a success
        if success_match and not failure_match:
            return True
        
        # If failure pattern found and no success pattern, it's a failure
        if failure_match and not success_match:
            return False
            
        # Check if both or neither patterns found, check "yes" or "no" at beginning
        if text.startswith("yes"):
            return True
        elif text.startswith("no"):
            return False
            
        # If still undetermined, count success and failure mentions
        success_count = 0
        failure_count = 0
        
        for pattern in success_patterns:
            success_count += len(re.findall(pattern, text))
            
        for pattern in failure_patterns:
            failure_count += len(re.findall(pattern, text))
        
        if success_count > failure_count:
            return True
        elif failure_count > success_count:
            return False
            
        # If still undetermined, return None
        print(f"Warning: Could not determine prediction from response:\n{response_text}")
        return None
        
    def extract_explanation(self, response_text):
        """
        从模型响应中提取解释文本
        
        Args:
            response_text: 模型的响应文本
            
        Returns:
            explanation: 模型的解释
        """
        # 简单地返回整个响应作为解释
        # 如果需要，可以实现更复杂的提取逻辑
        return response_text
    
    def evaluate_trial(self, trial_path: Path, shot_count: int) -> Dict:
        """评估单个试验"""
        print(f"评估: {trial_path.name}, {shot_count}-shot")
        
        messages, true_result, game_type = self.build_prompt(trial_path, shot_count)
        if messages is None:  # 检查是否应该跳过此试验
            return None
        
        prediction, explanation, response_time = self.call_model(messages)
        if prediction is None:
            return {
                "trial": trial_path.name,
                "shot_count": shot_count,
                "game_type": game_type,
                "true_result": true_result,
                "prediction": None,
                "correct": None,
                "response": None,
                "confidence": None,
                "response_time": response_time
            }
        
        # 简单计算置信度（基于关键词出现的次数和位置）
        confidence = None  # 可以根据需要实现置信度计算
        
        result = {
            "trial": trial_path.name,
            "shot_count": shot_count,
            "game_type": game_type,
            "true_result": true_result,
            "prediction": prediction,
            "Result": prediction == true_result,
            "response": explanation,
            "confidence": confidence,
            "response_time": response_time
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
    
    def prepare_one_shot_examples(self, test_trial: Path, strategy: str, used_examples: set = None) -> List[Dict]:
        """
        Prepare 1-shot examples that only contain the initial frame.
        
        Args:
            test_trial: Test trial path (Path object)
            strategy: Example selection strategy ('success' or 'failure')
            used_examples: Set of already used examples in the current trial (ignored as examples can be reused)
        
        Returns:
            List[Dict]: List containing one example with initial frame and result
        """
        # Get game type
        game_type = test_trial.name.split('_')[0]
        # Extract base game type (remove potential A or B suffixes)
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        # Filter trials with the same game type as the test trial
        candidates = []
        for trial_folder in self.find_trial_folders():
            # Ensure we don't use the test trial itself as an example
            if trial_folder == test_trial:
                continue
            
            # Check if trial type matches
            trial_game_type = trial_folder.name.split('_')[0]
            trial_base_type = trial_game_type.split('A')[0].split('B')[0]
            
            # Check if success/failure status matches
            is_success = trial_folder.name.endswith("True")
            if (strategy == 'success' and is_success) or (strategy == 'failure' and not is_success):
                # Prioritize trials with exact game type match
                if trial_game_type == game_type:
                    candidates.append(trial_folder)
                # Then consider trials with matching base game type
                elif trial_base_type == base_game_type:
                    candidates.append(trial_folder)
                    
        # If no candidate trials found, issue a warning and return empty list
        if not candidates:
            print(f"Warning: Could not find examples matching {game_type} with strategy {strategy}")
            return []
        
        # Randomly select a candidate trial
        selected_trial = random.choice(candidates)
        print(f"    Selected example: {selected_trial.name}")
        
        # Create example
        try:
            # Get initial frame
            initial_frame = next(selected_trial.glob("frame_0000.png"))
            
            # Determine success status
            is_success = selected_trial.name.endswith("True")
            
            # Create example
            example = {
                "trial": str(selected_trial),
                "game_type": selected_trial.name.split('_')[0],
                "initial_frame": str(initial_frame),
                "is_success": is_success
            }
            
            return [example]
            
        except StopIteration:
            print(f"Warning: Could not find initial frame in {selected_trial}")
            return []
    
    def prepare_two_shot_examples(self, test_trial: Path, first_shot_examples: List[Dict], strategy: str, used_examples: set = None) -> List[Dict]:
        """
        Prepare 2-shot examples that only contain the initial frame.
        
        Args:
            test_trial: Test trial path (Path object)
            first_shot_examples: List of first shot examples
            strategy: Example selection strategy ('success_success', 'success_failure', 'failure_success', or 'failure_failure')
            used_examples: Set of already used examples in the current trial (ignored as examples can be reused)
            
        Returns:
            List[Dict]: List containing two examples, each with initial frame and result
        """
        # If there's no first example, return empty list
        if not first_shot_examples:
            return []
            
        # Get game type
        game_type = test_trial.name.split('_')[0]
        # Extract base game type (remove potential A or B suffixes)
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        # Determine if second example should be success or failure
        if strategy == 'success_success' or strategy == 'failure_success':
            second_strategy = 'success_example'
        else:
            second_strategy = 'failure_example'
            
        # Filter trials with the same game type as the test trial
        candidates = []
        for trial_folder in self.find_trial_folders():
            # Ensure we don't use the test trial itself as an example
            if trial_folder == test_trial:
                continue
                
            # Ensure we don't use trials already in the first shot examples
            if any(example["trial"] == str(trial_folder) for example in first_shot_examples):
                continue
                
            # Check if trial type matches
            trial_game_type = trial_folder.name.split('_')[0]
            trial_base_type = trial_game_type.split('A')[0].split('B')[0]
            
            # Check if success/failure status matches
            is_success = trial_folder.name.endswith("True")
            if (second_strategy == 'success_example' and is_success) or (second_strategy == 'failure_example' and not is_success):
                # Prioritize trials with exact game type match
                if trial_game_type == game_type:
                    candidates.append(trial_folder)
                # Then consider trials with matching base game type
                elif trial_base_type == base_game_type:
                    candidates.append(trial_folder)
                    
        # If no candidate trials found, issue a warning and return first shot examples
        if not candidates:
            print(f"Warning: Could not find second example matching {game_type} with strategy {second_strategy}")
            return first_shot_examples
            
        # Randomly select a candidate trial
        selected_trial = random.choice(candidates)
        print(f"    Selected second example: {selected_trial.name}")
        
        # Create second example
        try:
            # Get initial frame
            initial_frame = next(selected_trial.glob("frame_0000.png"))
            
            # Determine success status
            is_success = selected_trial.name.endswith("True")
            
            # Create example
            second_example = {
                "trial": str(selected_trial),
                "game_type": selected_trial.name.split('_')[0],
                "initial_frame": str(initial_frame),
                "is_success": is_success
            }
            
            # Add second example to list
            examples = first_shot_examples.copy()
            examples.append(second_example)
            
            return examples
        
        except StopIteration:
            print(f"Warning: Could not find initial frame in {selected_trial}")
            return first_shot_examples
    
    def prepare_three_shot_examples(self, test_trial: Path, two_shot_examples: List[Dict], strategy: str, used_examples: set = None) -> List[Dict]:
        """
        Prepare 3-shot examples that only contain the initial frame.
        
        Args:
            test_trial: Test trial path (Path object)
            two_shot_examples: List of two shot examples
            strategy: Example selection strategy ('success' or 'failure')
            used_examples: Set of already used examples in the current trial (ignored as examples can be reused)
            
        Returns:
            List[Dict]: List containing three examples, each with initial frame and result
        """
        # If there aren't two examples, return what we have
        if len(two_shot_examples) < 2:
            return two_shot_examples
            
        # Get game type
        game_type = test_trial.name.split('_')[0]
        # Extract base game type (remove potential A or B suffixes)
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        # Filter trials with the same game type as the test trial
        candidates = []
        for trial_folder in self.find_trial_folders():
            # Ensure we don't use the test trial itself as an example
            if trial_folder == test_trial:
                continue
                
            # Ensure we don't use trials already in the two shot examples
            if any(example["trial"] == str(trial_folder) for example in two_shot_examples):
                continue
                
            # Check if trial type matches
            trial_game_type = trial_folder.name.split('_')[0]
            trial_base_type = trial_game_type.split('A')[0].split('B')[0]
            
            # Check if success/failure status matches
            is_success = trial_folder.name.endswith("True")
            if (strategy == 'success' and is_success) or (strategy == 'failure' and not is_success):
                # Prioritize trials with exact game type match
                if trial_game_type == game_type:
                    candidates.append(trial_folder)
                # Then consider trials with matching base game type
                elif trial_base_type == base_game_type:
                    candidates.append(trial_folder)
                    
        # If no candidate trials found, issue a warning and return two shot examples
        if not candidates:
            print(f"Warning: Could not find third example matching {game_type} with strategy {strategy}")
            return two_shot_examples
            
        # Randomly select a candidate trial
        selected_trial = random.choice(candidates)
        print(f"    Selected third example: {selected_trial.name}")
        
        # Create third example
        try:
            # Get initial frame
            initial_frame = next(selected_trial.glob("frame_0000.png"))
            
            # Determine success status
            is_success = selected_trial.name.endswith("True")
            
            # Create example
            third_example = {
                "trial": str(selected_trial),
                "game_type": selected_trial.name.split('_')[0],
                "initial_frame": str(initial_frame),
                "is_success": is_success
            }
            
            # Add third example to list
            examples = two_shot_examples.copy()
            examples.append(third_example)
            
            return examples
        
        except StopIteration:
            print(f"Warning: Could not find initial frame in {selected_trial}")
            return two_shot_examples
    
    def build_prompt_with_examples(self, test_trial: Path, examples: List[Dict]) -> str:
        """
        Build a prompt with given examples.
        
        Args:
            test_trial: Test trial path
            examples: List of examples, each containing initial frame and result
            
        Returns:
            str: Prompt with examples
        """
        # Get test trial's first frame
        try:
            first_frame = next(test_trial.glob("frame_0000.png"))
        except StopIteration:
            print(f"Warning: Could not find first frame in {test_trial}")
            return None
        
        # Determine true result
        true_result = test_trial.name.endswith("True")
        
        # Extract game type from trial name
        game_type = test_trial.name.split('_')[0]
        # Extract base game type (remove potential A or B suffixes)
        base_game_type = game_type.split('A')[0].split('B')[0]
        
        # Use scenario-specific prompt or general prompt
        if game_type in self.scenario_prompts:
            system_prompt = self.scenario_prompts[game_type]
        elif base_game_type in self.scenario_prompts:
            system_prompt = self.scenario_prompts[base_game_type]
        else:
            system_prompt = self.scenario_prompts.get("default", "")
            
        # Prepare messages
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": system_prompt})
        
        # Add examples (if any)
        for example in examples:
            # Use user prompt to show initial scene
            user_content = "This is the initial scene. Will the red ball successfully reach the green target?"
            
            # Read and encode initial frame
            with open(example["initial_frame"], "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                
            # Add user message (with image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            })
            
            # Add assistant reply (based on example's success/failure)
            if example["is_success"]:
                assistant_content = "Yes, the red ball will successfully reach the green target."
            else:
                assistant_content = "No, the red ball will not reach the green target."
                
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
            
        # Add test question
        user_content = "This is the initial scene. Will the red ball successfully reach the green target?"
        
        # Read and encode test's initial frame
        with open(first_frame, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            
        # Add test question (with image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        })
        
        return messages
    
    def run_controlled_shot_experiment(self, num_trials=10, temperature=0.0, seed=42, result_dir="results", balance=False, use_all_trials=True):
        """
        Run a controlled experiment with different n-shot strategies
        """
        np.random.seed(seed)
        result_dir = Path(result_dir)
        result_dir.mkdir(exist_ok=True)
        
        # Select trials
        if use_all_trials:
            print(f"Using all trials from the dataset")
            selected_trials = self.find_trial_folders()
        else:
            print(f"Selecting {num_trials} diverse trials{', balancing success and failure cases' if balance else ''}")
            selected_trials = self.select_diverse_trials(num_trials, balance_success=balance)
        
        # Record selected trials
        selected_trials_file = result_dir / "selected_trials.txt"
        with open(selected_trials_file, "w") as f:
            for trial in selected_trials:
                f.write(f"{trial}\n")
        
        # Define phase 1 conditions (0-shot, 1-shot, and four 2-shot strategies)
        phase1_conditions = ["0shot", "success_1shot", "failure_1shot", "success_success_2shot", "success_failure_2shot", "failure_success_2shot", "failure_failure_2shot"]
        
        results = []
        
        for i, trial_path in enumerate(selected_trials):
            print(f"Processing trial {i+1}/{len(selected_trials)}: {trial_path.name}")
            
            # Extract true result from trial name
            trial_name = trial_path.name
            true_result = "True" in trial_name
            
            # Extract game type from trial name
            game_type = trial_name.split("_")[0]
            
            # Get frame files
            frame_files = sorted(list(trial_path.glob("*.png")))
            if not frame_files:
                print(f"Warning: No frame files found for trial {trial_path.name}")
                continue
            
            # Prepare experiments for different conditions
            for condition in phase1_conditions:
                # Prepare examples
                examples = []
                
                if condition == "0shot":
                    # 0-shot: no examples
                    pass
                elif condition.endswith("1shot"):
                    # 1-shot: success or failure example
                    strategy = condition.split("_")[0]  # "success" or "failure"
                    examples = self.prepare_one_shot_examples(trial_path, strategy)
                    if not examples:
                        print(f"Warning: Could not find examples for {condition} condition")
                        continue
                elif condition.endswith("2shot"):
                    # 2-shot: combination of strategies
                    strategy_parts = condition.split("_")
                    strategy1 = strategy_parts[0]  # First example strategy (success or failure)
                    strategy2 = strategy_parts[1]  # Second example strategy (success or failure)
                    
                    # First get one example
                    one_shot_examples = self.prepare_one_shot_examples(trial_path, strategy1)
                    if not one_shot_examples:
                        print(f"Warning: Could not find first example for {condition} condition")
                        continue
                
                    # Then get two examples
                    examples = self.prepare_two_shot_examples(trial_path, one_shot_examples, strategy2)
                    if len(examples) < 2:
                        print(f"Warning: Could not find enough examples for {condition} condition")
                        continue
                
                # Build prompt with examples
                messages = self.build_prompt_with_examples(trial_path, examples)
                if not messages:
                    continue
                
                # Call model and get response
                prediction, explanation, response_time = self.call_model(messages, temperature)
                
                # Record results
                trial_result = {
                    "trial": trial_path.name,
                    "condition": condition,
                    "true_result": true_result,
                    "prediction": prediction,
                    "correct": prediction == true_result if prediction is not None else None,
                    "explanation": explanation,
                    "game_type": game_type,
                    "response_time": response_time
                }
                
                results.append(trial_result)
                print(f"  Condition {condition}: Prediction = {prediction}, Correct = {trial_result['correct']}")
        
        # Save results
        self.results = results
        
        # Analyze 2-shot strategy results to determine best strategy
        df = pd.DataFrame(results)
        two_shot_df = df[df["condition"].str.endswith("2shot")]
        
        # Calculate accuracy for each 2-shot strategy
        strategy_accuracy = two_shot_df.groupby("condition")["correct"].mean()
        sorted_strategies = strategy_accuracy.sort_values(ascending=False)
        
        print("\n2-shot strategy accuracy:")
        print(sorted_strategies)
        
        best_strategy = sorted_strategies.index[0] if not sorted_strategies.empty else None
        second_best_strategy = sorted_strategies.index[1] if len(sorted_strategies) > 1 else None
        
        if best_strategy and second_best_strategy:
            print(f"\nBest 2-shot strategy: {best_strategy}, Accuracy: {sorted_strategies[0]:.2f}")
            print(f"Second best 2-shot strategy: {second_best_strategy}, Accuracy: {sorted_strategies[1]:.2f}")
            
            # Phase 2: Create 3-shot experiments using best 2-shot strategies
            conditions_map = {
                best_strategy + "+success": "best_success_3shot",
                best_strategy + "+failure": "best_failure_3shot",
                second_best_strategy + "+success": "second_best_success_3shot",
                second_best_strategy + "+failure": "second_best_failure_3shot"
            }
            
            # Convert 2-shot strategy names to lists
            best_strategy_parts = best_strategy.split("_")[:2]  # Get strategy parts (e.g., ["success", "failure"])
            second_best_strategy_parts = second_best_strategy.split("_")[:2]
            
            for i, trial_path in enumerate(selected_trials):
                print(f"Phase 2 - Processing trial {i+1}/{len(selected_trials)}: {trial_path.name}")
                
                # Extract true result from trial name
                trial_name = trial_path.name
                true_result = "True" in trial_name
                
                # Extract game type from trial name
                game_type = trial_name.split("_")[0]
                
                # Get frame files
                frame_files = sorted(list(trial_path.glob("*.png")))
                if not frame_files:
                    continue
                    
                # Prepare 3-shot experiments for best and second-best 2-shot strategies
                for base_strategy, strategy_parts in [(best_strategy, best_strategy_parts), (second_best_strategy, second_best_strategy_parts)]:
                    for third_example_strategy in ["success", "failure"]:
                        # Get condition name
                        condition = conditions_map[base_strategy + "+" + third_example_strategy]
                        
                        # First get one example
                        one_shot_examples = self.prepare_one_shot_examples(trial_path, strategy_parts[0])
                        if not one_shot_examples:
                            print(f"Warning: Could not find first example for {condition} condition")
                            continue
                        
                        # Then get two examples
                        two_shot_examples = self.prepare_two_shot_examples(trial_path, one_shot_examples, strategy_parts[1])
                        if len(two_shot_examples) < 2:
                            print(f"Warning: Could not find enough examples for {condition} condition")
                            continue
                        
                        # Finally get three examples
                        examples = self.prepare_three_shot_examples(trial_path, two_shot_examples, third_example_strategy)
                        if len(examples) < 3:
                            print(f"Warning: Could not find enough examples for {condition} condition")
                            continue
                        
                        # Build prompt with examples
                        messages = self.build_prompt_with_examples(trial_path, examples)
                        if not messages:
                            continue
                        
                        # Call model and get response
                        prediction, explanation, response_time = self.call_model(messages, temperature)
                        
                        # Record results
                        trial_result = {
                            "trial": trial_path.name,
                            "condition": condition,
                            "true_result": true_result,
                            "prediction": prediction,
                            "correct": prediction == true_result if prediction is not None else None,
                            "explanation": explanation,
                            "game_type": game_type,
                            "response_time": response_time
                        }
                        
                        results.append(trial_result)
                        print(f"  Condition {condition}: Prediction = {prediction}, Correct = {trial_result['correct']}")
        
        # Update and save all results
        self.results = results
        
        # Analyze and save results
        self.analyze_and_save_results(result_dir)
    
    def analyze_and_save_results(self, result_dir: Path):
        """
        Analyze the results of the controlled experiment and save them to files
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
        
        # Calculate average response time
        overall_response_time = df['response_time'].mean()
        print(f"\nAverage response time: {overall_response_time:.4f} seconds")
        
        # 1. Analyze by condition
        condition_analysis = df.groupby('condition').agg({
            'correct': ['mean', 'count'],
            'response_time': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        condition_analysis.columns = ['_'.join(col).strip('_') for col in condition_analysis.columns.values]
        
        # Rename columns for clarity
        condition_analysis = condition_analysis.rename(columns={
            'condition_': 'Condition',
            'correct_mean': 'Accuracy',
            'correct_count': 'Count',
            'response_time_mean': 'Avg_Response_Time',
            'response_time_std': 'Std_Response_Time',
            'response_time_min': 'Min_Response_Time',
            'response_time_max': 'Max_Response_Time'
        })
        
        # Sort by accuracy
        condition_analysis = condition_analysis.sort_values('Accuracy', ascending=False)
        
        print("\nAccuracy and response time by condition (sorted by accuracy):")
        for i, row in condition_analysis.iterrows():
            print(f"  {row['Condition']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})")
        
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
        shot_analysis = df.groupby('shot_count').agg({
            'correct': ['mean', 'count'],
            'response_time': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        shot_analysis.columns = ['_'.join(col).strip('_') for col in shot_analysis.columns.values]
        
        # Rename columns for clarity
        shot_analysis = shot_analysis.rename(columns={
            'shot_count_': 'Shot_Count',
            'correct_mean': 'Accuracy',
            'correct_count': 'Count',
            'response_time_mean': 'Avg_Response_Time',
            'response_time_std': 'Std_Response_Time',
            'response_time_min': 'Min_Response_Time',
            'response_time_max': 'Max_Response_Time'
        })
        
        shot_analysis = shot_analysis.sort_values('Shot_Count')
        
        print("\nAccuracy and response time by shot count:")
        for i, row in shot_analysis.iterrows():
            print(f"  {int(row['Shot_Count'])}-shot: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})")
        
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
        pattern_analysis = df.groupby(['shot_count', 'example_pattern']).agg({
            'correct': ['mean', 'count'],
            'response_time': ['mean', 'std']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        pattern_analysis.columns = ['_'.join(col).strip('_') for col in pattern_analysis.columns.values]
        
        # Rename columns for clarity
        pattern_analysis = pattern_analysis.rename(columns={
            'shot_count_': 'Shot_Count',
            'example_pattern_': 'Example_Pattern',
            'correct_mean': 'Accuracy',
            'correct_count': 'Count',
            'response_time_mean': 'Avg_Response_Time',
            'response_time_std': 'Std_Response_Time'
        })
        
        pattern_analysis = pattern_analysis.sort_values(['Shot_Count', 'Accuracy'], ascending=[True, False])
        
        print("\nAccuracy and response time by example pattern:")
        for sc in sorted(pattern_analysis['Shot_Count'].unique()):
            print(f"  {int(sc)}-shot patterns:")
            subset = pattern_analysis[pattern_analysis['Shot_Count'] == sc]
            for i, row in subset.iterrows():
                print(f"    {row['Example_Pattern']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})")
        
        # 4. Analyze by game type
        game_type_analysis = df.groupby(['game_type']).agg({
            'correct': 'mean',
            'response_time': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        game_type_analysis.columns = ['_'.join(col).strip('_') for col in game_type_analysis.columns.values]
        
        # Rename columns for clarity
        game_type_analysis = game_type_analysis.rename(columns={
            'game_type_': 'Game_Type',
            'correct_': 'Accuracy',
            'response_time_mean': 'Avg_Response_Time',
            'response_time_std': 'Std_Response_Time',
            'response_time_count': 'Count'
        })
        
        game_type_analysis = game_type_analysis.sort_values('Accuracy', ascending=False)
        
        print("\nAccuracy and response time by game type:")
        for i, row in game_type_analysis.iterrows():
            print(f"  {row['Game_Type']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})")
        
        # Save detailed numeric results to text file
        with open(result_dir / 'controlled_experiment_analysis.txt', 'w') as f:
            f.write("===== CONTROLLED EXPERIMENT ANALYSIS =====\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")
            f.write(f"Average response time: {overall_response_time:.4f} seconds\n\n")
            
            f.write("Accuracy and response time by condition:\n")
            for i, row in condition_analysis.iterrows():
                f.write(f"  {row['Condition']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy and response time by shot count:\n")
            for i, row in shot_analysis.iterrows():
                f.write(f"  {int(row['Shot_Count'])}-shot: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy and response time by example pattern:\n")
            for sc in sorted(pattern_analysis['Shot_Count'].unique()):
                f.write(f"  {int(sc)}-shot patterns:\n")
                subset = pattern_analysis[pattern_analysis['Shot_Count'] == sc]
                for i, row in subset.iterrows():
                    f.write(f"    {row['Example_Pattern']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})\n")
            
            f.write("\nAccuracy and response time by game type:\n")
            for i, row in game_type_analysis.iterrows():
                f.write(f"  {row['Game_Type']}: Acc={row['Accuracy']:.4f}, Time={row['Avg_Response_Time']:.2f}s ±{row['Std_Response_Time']:.2f}s (n={int(row['Count'])})\n")
            
            f.write("\nDetailed condition by game type analysis:\n")
            f.write(df.pivot_table(
                index='game_type', 
                columns='condition', 
                values=['correct', 'response_time'],
                aggfunc={'correct': 'mean', 'response_time': 'mean'}
            ).to_string())
        
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
        condition_df = condition_analysis.copy()
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
        
        plt.savefig(result_dir / 'accuracy_by_condition.png', dpi=300, bbox_inches='tight')
        
        # Plot response time by condition
        plt.figure(figsize=(12, 6))
        bars = plt.bar(condition_df['Condition'], condition_df['Avg_Response_Time'], color='lightgreen')
        
        # Color-code bars by shot count
        for i, bar in enumerate(bars):
            condition = condition_df.iloc[i]['Condition']
            shot_count = get_shot_count(condition)
            bar.set_color(colors[shot_count])
        
        plt.title('Response Time by Experimental Condition')
        plt.xlabel('Condition')
        plt.ylabel('Response Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}s', ha='center', va='bottom', rotation=0)
        
        # Add legend
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(result_dir / 'response_time_by_condition.png', dpi=300, bbox_inches='tight')
        
        # Plot accuracy by shot count
        plt.figure(figsize=(8, 6))
        plt.bar(shot_analysis['Shot_Count'].astype(str) + '-shot', shot_analysis['Accuracy'], color=['#FFC107', '#4CAF50', '#2196F3', '#9C27B0'])
        plt.title('Accuracy by Shot Count')
        plt.xlabel('Shot Count')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)  # Chance level
        
        # Add value labels
        for i, v in enumerate(shot_analysis['Accuracy']):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.savefig(result_dir / 'accuracy_by_shot_count.png', dpi=300, bbox_inches='tight')
        
        # Plot response time by shot count
        plt.figure(figsize=(8, 6))
        plt.bar(shot_analysis['Shot_Count'].astype(str) + '-shot', shot_analysis['Avg_Response_Time'], color=['#FFC107', '#4CAF50', '#2196F3', '#9C27B0'])
        plt.title('Response Time by Shot Count')
        plt.xlabel('Shot Count')
        plt.ylabel('Response Time (seconds)')
        
        # Add value labels
        for i, v in enumerate(shot_analysis['Avg_Response_Time']):
            plt.text(i, v + 0.02, f'{v:.2f}s', ha='center', va='bottom')
        
        plt.savefig(result_dir / 'response_time_by_shot_count.png', dpi=300, bbox_inches='tight')
        
        # Plot success vs. failure vs. mixed examples effect
        plt.figure(figsize=(10, 8))
        
        # Filter to just show 1-shot and 2-shot for pattern comparison
        pattern_subset = pattern_analysis[pattern_analysis['Shot_Count'].isin([1, 2])]
        pattern_subset = pattern_subset.sort_values(['Shot_Count', 'Accuracy'], ascending=[True, False])
        
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
            pattern = pattern_subset.iloc[i]['Example_Pattern']
            if pattern in pattern_colors:
                bar.set_color(pattern_colors[pattern])
        
        plt.title('Accuracy by Example Pattern Type')
        plt.xlabel('Example Pattern')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)  # Chance level
        
        # Customize x-axis
        plt.xticks(indices, [f"{int(row['Shot_Count'])}-shot: {row['Example_Pattern']}" 
                             for _, row in pattern_subset.iterrows()], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(result_dir / 'accuracy_by_pattern.png', dpi=300, bbox_inches='tight')
        
        # Plot response time by pattern
        plt.figure(figsize=(10, 8))
        bars = plt.bar(indices, pattern_subset['Avg_Response_Time'])
        
        for i, bar in enumerate(bars):
            pattern = pattern_subset.iloc[i]['Example_Pattern']
            if pattern in pattern_colors:
                bar.set_color(pattern_colors[pattern])
                
        plt.title('Response Time by Example Pattern Type')
        plt.xlabel('Example Pattern')
        plt.ylabel('Response Time (seconds)')
        
        # Customize x-axis
        plt.xticks(indices, [f"{int(row['Shot_Count'])}-shot: {row['Example_Pattern']}" 
                             for _, row in pattern_subset.iterrows()], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(result_dir / 'response_time_by_pattern.png', dpi=300, bbox_inches='tight')
        
        # Create combined visualization - accuracy vs response time
        plt.figure(figsize=(10, 8))
        plt.scatter(condition_analysis['Avg_Response_Time'], condition_analysis['Accuracy'], 
                   c=[colors[get_shot_count(cond)] for cond in condition_analysis['Condition']], 
                   s=100, alpha=0.7)
        
        # Add condition labels
        for i, row in condition_analysis.iterrows():
            plt.annotate(row['Condition'], 
                        (row['Avg_Response_Time'], row['Accuracy']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.title('Accuracy vs. Response Time by Condition')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(result_dir / 'accuracy_vs_response_time.png', dpi=300, bbox_inches='tight')
        
        print("\nAnalysis complete. Results saved to the output directory.")
        return condition_analysis

def main():
    """
    Main entry point for the physical intuition evaluator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--model_name", type=str, default="qwen2.5-omni-7b", help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base URL")
    parser.add_argument("--output_dir", type=str, default="results1", help="Output directory")
    parser.add_argument("--prompt_dir", type=str, default="prompt1", help="Prompt directory")
    parser.add_argument("--result_folder", type=str, default=None, help="Result folder name")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials (only used if use_all_trials=False)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter")
    parser.add_argument("--use_all_trials", action="store_true", default=True, help="Use all available trials (ignores num_trials parameter)")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create evaluator
    evaluator = PhysicalIntuitionEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        result_folder=args.result_folder
    )
    
    # Run controlled experiment
    evaluator.run_controlled_shot_experiment(
        num_trials=args.num_trials, 
        temperature=args.temperature,
        use_all_trials=args.use_all_trials
    )

if __name__ == "__main__":
    main()