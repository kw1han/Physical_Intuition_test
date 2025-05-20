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
import re
from datetime import datetime
import requests
#DS
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

class PhysicalIntuitionEvaluator:
    def __init__(self, 
                 data_root: str, 
                 model_name: str = "gemma3:27b",
                 api_key: Optional[str] = None,
                 base_url: str = "http://localhost:11434/v1",
                 output_dir: str = "results",
                 game_type: Optional[str] = None,
                 max_tokens: int = 512,
                 temperature: float = 0.6,
                 top_p: float = 0.9):
        """
        初始化物理直觉评估器
        
        Args:
            data_root: 数据根目录
            model_name: 要测试的模型名称
            api_key: API密钥
            base_url: API基础URL
            output_dir: 结果输出目录
            game_type: 要测试的游戏类型，如果为None则测试所有类型
            max_tokens: 生成回复的最大token数
            temperature: 控制回复的随机性
            top_p: 控制多样性的核采样参数
        """
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.game_type = game_type  # 保存游戏类型
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # 添加提示文件的路径
        self.prompt_dir = Path("/home/student0/Physical_Intuition_test/prompt")
        
        # 初始化日志记录器
        self.logger = Logger(self.output_dir)
        
        # 设置API密钥，优先使用传入的参数，其次使用环境变量
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-uruhfkcrzkebvehlyoeradzzwentjyjpqbteiryddkkpahpe"
        
        # 初始化API客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            # 确保Authorization头部格式正确
            default_headers={
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
        # 初始化结果记录
        self.results = []
        self.response_times = []  # 记录响应时间
        
        # 获取所有主题文件夹
        self.subjects = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("Subj_")])
        
        # 游戏类型映射到中文描述（用于提示）
        self.game_type_desc = {
            "Basic": "Basic Physical Scene",
            "Bridge": "Bridge Building Scene",
            "Catapult": "Catapult Scene",
            "Chaining": "Chain Reaction Scene",
            "Falling": "Object Falling Scene",
            "Gap": "Gap Crossing Scene",
            "Launch": "Launching Scene",
            "Prevention": "Motion Prevention Scene",
            "SeeSaw": "Seesaw Scene",
            "Shafts": "Shaft Scene",
            "Table": "Table Scene",
            "Unbox": "Unboxing Scene",
            "Unsupport": "Support Removal Scene"
        }
    
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
    
    def call_model(self, messages: List[Dict]) -> Dict:
        """
        调用模型
        
        Args:
            messages: 消息列表
            
        Returns:
            模型响应
        """
        max_retries = 3
        retry_delay = 3  # 初始延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                self.logger.log(f"\nAttempt {attempt+1}/{max_retries} to call API with model: {self.model_name}")
                self.logger.log(f"Using API URL: {self.client.base_url}")
                
                # 第一次尝试时打印更多信息
                if attempt == 0:
                    self.logger.log(f"API key (first 5 chars): {self.client.api_key[:5]}...")
                    self.logger.log(f"Number of messages: {len(messages)}")
                    self.logger.log(f"First message role: {messages[0]['role']}")
                
                # 构建请求参数
                request_data = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": False  # 不使用流式响应
                }
                
                # 使用client发送请求
                response = self.client.chat.completions.create(**request_data)
                self.logger.log("API call successful!")
                return response
                
            except Exception as e:
                error_msg = str(e)
                self.logger.log(f"API调用错误 (Attempt {attempt+1}/{max_retries}): {error_msg}")
                
                # 检查错误类型
                if "invalid_api_key" in error_msg or "401" in error_msg:
                    self.logger.log("API密钥无效。请检查API密钥是否正确，或尝试使用环境变量OPENAI_API_KEY设置密钥。")
                    if attempt == max_retries - 1:  # 最后一次尝试
                        self.logger.log("API密钥问题仍然存在，建议：")
                        self.logger.log("1. 检查API密钥是否已过期")
                        self.logger.log("2. 确认API密钥格式是否正确")
                        self.logger.log("3. 尝试使用环境变量设置: export OPENAI_API_KEY='your-key'")
                        self.logger.log("4. 确认服务器URL与API密钥匹配")
                elif "rate_limit" in error_msg:
                    self.logger.log("达到API速率限制，将增加重试延迟时间")
                    retry_delay *= 2  # 指数退避
                
                # 等待一段时间再重试
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    self.logger.log(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
        
        self.logger.log("所有重试尝试都失败，返回None")
        return None
    
    
    def run_evaluation(self, num_sets: int = 5, repetitions: int = 4):
        """
        运行物理直觉评估
        
        Args:
            num_sets: 已弃用，保留参数是为了兼容性
            repetitions: 对每组图片重复测试的次数，用于验证模型分析的一致性
        """
        # 重置结果
        self.results = []
        self.response_times = []
        
        # 获取同一场景下的成功和失败试验
        valid_scenes = self.find_same_scene_trials()
        
        if not valid_scenes:
            self.logger.log("No valid scenes found (require at least 3 failure cases and 1 success case in the same scene)")
            return
        
        # 对每个场景的每个成功案例进行测试
        for failure_trials, success_trials in valid_scenes:
            self.logger.log(f"\nStarting evaluation for scene: {success_trials[0].name.split('_')[0]}_{success_trials[0].name.split('_')[3]}")
            self.logger.log(f"This scene has {len(success_trials)} success cases")
            
            # 获取当前场景的游戏类型
            current_game_type = success_trials[0].name.split('_')[0]
            
            # 对每个成功案例进行测试
            for success_case in success_trials:
                # 随机选择3个失败案例
                selected_failures = random.sample(failure_trials, 3)
                
                # 组合所有案例
                all_cases = selected_failures + [success_case]
                
                # 检查是否所有案例都有效
                valid_cases = []
                valid_images = []
                for case in all_cases:
                    try:
                        first_frame = next(case.glob("frame_0000.png"))
                        base64_img = encode_image_to_base64(str(first_frame))
                        valid_images.append(base64_img)
                        valid_cases.append(case)
                    except (StopIteration, FileNotFoundError) as e:
                        self.logger.log(f"Warning: Could not find first frame image in {case}, skipping this case")
                
                # 如果没有足够的有效案例，跳过这个组合
                if len(valid_cases) < 4:
                    self.logger.log(f"Warning: Not enough valid cases in the current combination (need 4), skipping")
                    continue
                
                # 更新所有案例列表
                all_cases = valid_cases
                original_correct_index = all_cases.index(success_case) + 1
                
                # 记录该组图像一致性测试的结果
                consistency_results = []
                
                # 对该组图像进行多次重复测试（每次重新打乱顺序）
                self.logger.log(f"\n----- Starting {repetitions} repetition tests for the same image set -----")
                
                # 确保正确答案在四次测试中出现在不同位置 (A-B-C-D)
                target_positions = list(range(4))  # 0,1,2,3对应A,B,C,D
                
                # 如果repetitions超过4，后面的位置随机分配
                if repetitions > 4:
                    extra_positions = [random.randint(0, 3) for _ in range(repetitions - 4)]
                    target_positions.extend(extra_positions)
                
                # 确保前四次测试的正确答案位置均不相同
                random.shuffle(target_positions)
                
                for rep in range(repetitions):
                    # 为当前重复确定目标位置
                    target_position = target_positions[rep]  # 目标位置(0-3)对应A-D
                    
                    # 重新构建顺序，确保成功案例在目标位置
                    # 首先，从所有案例中移除成功案例
                    cases_without_success = [case for case in all_cases if case != success_case]
                    # 随机打乱失败案例
                    random.shuffle(cases_without_success)
                    
                    # 构建新的顺序，将成功案例放在指定位置
                    shuffled_cases = cases_without_success.copy()
                    shuffled_cases.insert(target_position, success_case)
                    
                    # 找到案例对应的图片
                    shuffled_images = []
                    for case in shuffled_cases:
                        idx = all_cases.index(case)
                        shuffled_images.append(valid_images[idx])
                    
                    # 找到成功案例的新索引
                    correct_index = shuffled_cases.index(success_case) + 1
                    
                    # 构建原始顺序到新顺序的映射
                    random_indices = []
                    for case in shuffled_cases:
                        random_indices.append(all_cases.index(case))
                    
                    # 定义字母数组
                    letters = ['A', 'B', 'C', 'D']
                    
                    # 从文件加载提示内容
                    prompt_text = self.load_prompt_from_file(current_game_type)
                    
                    # 构建提示消息
                    messages = []
                    system_message = {
                        "role": "system",
                        "content": "You are an AI assistant with strong physical intuition. You need to analyze four physical scene images in their initial states and determine which scene will allow the red ball to successfully reach the green target area. These scenes come from the same physical environment setup but with slightly different initial conditions. Carefully observe the details of object positions, orientations, and obstacle distributions in each scene. Important note: Scenes are labeled with letters A-D, and the order has no correlation with success probability. Please make judgments completely based on physical principles. Structure your answer with a detailed reasoning for each scene followed by a final result statement."
                    }
                    messages.append(system_message)
                    
                    # 构建包含四张图片的提示
                    prompt_content = [
                        {"type": "text", "text": prompt_text}
                    ]
                    
                    # 使用字母A-D标记场景
                    for i, base64_img in enumerate(shuffled_images):
                        prompt_content.extend([
                            {"type": "text", "text": f"Scene {letters[i]}:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                        ])
                    
                    messages.append({"role": "user", "content": prompt_content})
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 调用模型获取响应
                    response = self.call_model(messages)
                    
                    # 记录响应时间
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    if not response:
                        continue
                        
                    response_text = response.choices[0].message.content
                    
                    # 从响应中提取预测的场景编号
                    predicted_index = self.extract_predicted_scene_number(response_text)
                    
                    # 找到预测场景的原始索引（在重排前的位置）
                    original_predicted_index = -1
                    if predicted_index > 0:
                        # 从字母索引映射回原始数据集中的索引
                        predicted_case_index = random_indices[predicted_index - 1]
                        original_predicted_index = predicted_case_index + 1
                    
                    # 记录结果
                    result = {
                        "set_index": len(self.results) + 1,
                        "repetition": rep + 1,
                        "correct_index": correct_index,
                        "predicted_index": predicted_index,
                        "original_correct_index": original_correct_index,
                        "original_predicted_index": original_predicted_index,
                        "correct": predicted_index == correct_index,
                        "response": response_text,
                        "response_time": response_time,
                        "trial_names": [case.name for case in shuffled_cases],
                        "original_trial_names": [case.name for case in all_cases],
                        "game_type": all_cases[0].name.split('_')[0],
                        "scene_id": all_cases[0].name.split('_')[3],
                        "success_case": success_case.name,
                        "shuffle_mapping": dict(zip(range(1, 5), [i+1 for i in random_indices]))
                    }
                    
                    self.results.append(result)
                    consistency_results.append(result)
                    
                    # 打印结果
                    self.logger.log(f"\n----- Test Set {len(self.results)} (Repetition {rep+1}/{repetitions}) -----")
                    self.logger.log(f"Success case: {success_case.name}")
                    self.logger.log(f"Scene type: {self.game_type_desc.get(result['game_type'], result['game_type'])}")
                    self.logger.log(f"Scene ID: {result['scene_id']}")
                    self.logger.log(f"Correct scene position (A-D): {letters[correct_index-1]}")
                    self.logger.log(f"Shuffle mapping: {result['shuffle_mapping']}")
                    self.logger.log(f"Model prediction: {letters[predicted_index-1] if predicted_index > 0 else 'Invalid prediction'}")
                    self.logger.log(f"Prediction {'correct' if result['correct'] else 'incorrect'}")
                    self.logger.log(f"Response time: {response_time:.2f} seconds")
                    
                    # 添加图片路径输出
                    self.logger.log("\nImage paths:")
                    for i, case in enumerate(shuffled_cases):
                        frame_path = next(case.glob("frame_0000.png"))
                        self.logger.log(f"Scene {letters[i]}: {frame_path}")
                    self.logger.log("")
                    
                    # 保存中间结果
                    self.save_results("physical_intuition_results.json")
                
                # 计算一致性统计
                # 检查模型是否在每次重复中都选择了同一个原始图像
                original_predictions = [r['original_predicted_index'] for r in consistency_results if r['original_predicted_index'] > 0]
                
                if original_predictions:
                    is_consistent = all(p == original_predictions[0] for p in original_predictions)
                    most_common = max(set(original_predictions), key=original_predictions.count)
                    consistency_ratio = original_predictions.count(most_common) / len(original_predictions)
                    
                    self.logger.log("\n----- Consistency Analysis -----")
                    self.logger.log(f"Original predictions across repetitions: {original_predictions}")
                    self.logger.log(f"Is perfectly consistent: {is_consistent}")
                    self.logger.log(f"Most frequently predicted image: {most_common}")
                    self.logger.log(f"Consistency ratio: {consistency_ratio:.2%}")
                    self.logger.log(f"Correct image was: {original_correct_index}")
                    self.logger.log("-------------------------------\n")
        
        # 分析总体结果
        self.analyze_results()

    def analyze_results(self):
        """分析评估结果"""
        if not self.results:
            self.logger.log("No results to analyze")
            return
        
        total_sets = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['correct'])
        accuracy = correct_predictions / total_sets
        
        # 1. 基础准确率统计
        self.logger.log("\n===== Physical Intuition Evaluation Results =====")
        self.logger.log(f"Total test sets: {total_sets}")
        self.logger.log(f"Correct predictions: {correct_predictions}")
        self.logger.log(f"Overall accuracy: {accuracy:.2%}")
        
        # 2. 位置偏差分析
        position_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        correct_position_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        for result in self.results:
            if result['predicted_index'] > 0:  # 排除无效预测
                position_stats[result['predicted_index']] += 1
                if result['correct']:
                    correct_position_stats[result['predicted_index']] += 1
        
        self.logger.log("\nPosition Bias Analysis:")
        for pos in range(1, 5):
            total = position_stats[pos]
            correct = correct_position_stats[pos]
            if total > 0:
                self.logger.log(f"Position {pos}: Predicted {total} times ({total/sum(position_stats.values()):.1%}), "
                              f"Correct {correct} times ({correct/total:.1%} accuracy)")
        
        # 3. 一致性分析
        self.logger.log("\nConsistency Analysis:")
        # 按测试组分组
        test_groups = {}
        for result in self.results:
            group_key = (result['game_type'], result['scene_id'], result['success_case'])
            if group_key not in test_groups:
                test_groups[group_key] = []
            test_groups[group_key].append(result)
        
        # 计算每组测试的一致性
        consistency_scores = []
        for group_key, group_results in test_groups.items():
            # 收集原始预测（基于图像的实际内容，而不是位置）
            original_predictions = [r['original_predicted_index'] for r in group_results if r['original_predicted_index'] > 0]
            if not original_predictions:
                continue
                
            # 计算一致性指标
            most_common = max(set(original_predictions), key=original_predictions.count)
            consistency_ratio = original_predictions.count(most_common) / len(original_predictions)
            consistency_scores.append(consistency_ratio)
            
            self.logger.log(f"Group {group_key[0]}_{group_key[1]}: Consistency ratio: {consistency_ratio:.2%}")
        
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            self.logger.log(f"Average consistency across all groups: {avg_consistency:.2%}")
            perfect_consistent_groups = sum(1 for score in consistency_scores if score == 1.0)
            self.logger.log(f"Groups with perfect consistency: {perfect_consistent_groups}/{len(consistency_scores)} ({perfect_consistent_groups/len(consistency_scores):.1%})")
        
        # 4. 响应时间分析
        avg_response_time = sum(self.response_times) / len(self.response_times)
        min_response_time = min(self.response_times)
        max_response_time = max(self.response_times)
        self.logger.log(f"\nResponse Time Analysis:")
        self.logger.log(f"Average response time: {avg_response_time:.2f} seconds")
        self.logger.log(f"Minimum response time: {min_response_time:.2f} seconds")
        self.logger.log(f"Maximum response time: {max_response_time:.2f} seconds")
        
        # 5. 按游戏类型分析
        game_type_results = {}
        for result in self.results:
            game_type = result['game_type']
            if game_type not in game_type_results:
                game_type_results[game_type] = {'total': 0, 'correct': 0, 'times': []}
            game_type_results[game_type]['total'] += 1
            if result['correct']:
                game_type_results[game_type]['correct'] += 1
            game_type_results[game_type]['times'].append(result['response_time'])
        
        self.logger.log("\nAnalysis by Scene Type:")
        for game_type, stats in game_type_results.items():
            type_accuracy = stats['correct'] / stats['total']
            avg_time = sum(stats['times']) / len(stats['times'])
            self.logger.log(f"\n{self.game_type_desc.get(game_type, game_type)}:")
            self.logger.log(f"  Accuracy: {type_accuracy:.2%} ({stats['correct']}/{stats['total']})")
            self.logger.log(f"  Average response time: {avg_time:.2f} seconds")
        
        # 6. 错误分析
        error_analysis = self.analyze_error_patterns()
        self.logger.log("\nError Pattern Analysis:")
        for error_type, count in error_analysis.items():
            self.logger.log(f"{error_type}: {count} times")
        
        # 7. 保存详细分析结果
        self.save_detailed_analysis(game_type_results, test_groups)

    def save_detailed_analysis(self, game_type_results, test_groups=None):
        """保存详细分析结果"""
        analysis_file = self.output_dir / 'physical_intuition_analysis.txt'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            # 定义字母数组
            letters = ['A', 'B', 'C', 'D']
            
            # 1. 总体统计
            f.write("===== Overall Statistics =====\n")
            total_sets = len(self.results)
            correct_predictions = sum(1 for r in self.results if r['correct'])
            f.write(f"Total test sets: {total_sets}\n")
            f.write(f"Correct predictions: {correct_predictions}\n")
            f.write(f"Overall accuracy: {correct_predictions/total_sets:.2%}\n\n")
            
            # 2. 一致性分析
            f.write("===== Consistency Analysis =====\n")
            if test_groups:
                consistency_scores = []
                for group_key, group_results in test_groups.items():
                    original_predictions = [r['original_predicted_index'] for r in group_results if r['original_predicted_index'] > 0]
                    if not original_predictions:
                        continue
                    
                    most_common = max(set(original_predictions), key=original_predictions.count)
                    consistency_ratio = original_predictions.count(most_common) / len(original_predictions)
                    consistency_scores.append(consistency_ratio)
                    
                    f.write(f"Group {group_key[0]}_{group_key[1]}:\n")
                    f.write(f"  Original predictions: {original_predictions}\n")
                    f.write(f"  Most common prediction: {most_common}\n")
                    f.write(f"  Consistency ratio: {consistency_ratio:.2%}\n")
                    f.write(f"  Original correct index: {group_results[0]['original_correct_index']}\n\n")
                
                if consistency_scores:
                    avg_consistency = sum(consistency_scores) / len(consistency_scores)
                    f.write(f"Average consistency across all groups: {avg_consistency:.2%}\n")
                    perfect_consistent_groups = sum(1 for score in consistency_scores if score == 1.0)
                    f.write(f"Groups with perfect consistency: {perfect_consistent_groups}/{len(consistency_scores)} ({perfect_consistent_groups/len(consistency_scores):.1%})\n\n")
            
            # 3. 响应时间分析
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            f.write("===== Response Time Analysis =====\n")
            f.write(f"Average response time: {avg_response_time:.2f} seconds\n")
            f.write(f"Minimum response time: {min_response_time:.2f} seconds\n")
            f.write(f"Maximum response time: {max_response_time:.2f} seconds\n\n")
            
            # 4. 按物理场景类型分析
            f.write("===== Analysis by Scene Type =====\n")
            for game_type, stats in game_type_results.items():
                type_accuracy = stats['correct'] / stats['total']
                avg_time = sum(stats['times']) / len(stats['times'])
                f.write(f"\n{self.game_type_desc.get(game_type, game_type)}:\n")
                f.write(f"  Accuracy: {type_accuracy:.2%} ({stats['correct']}/{stats['total']})\n")
                f.write(f"  Average response time: {avg_time:.2f} seconds\n")
            
            # 5. 详细结果记录
            f.write("\n===== Detailed Results =====\n")
            for i, result in enumerate(self.results, 1):
                f.write(f"\n--- Test Set {i} (Rep {result.get('repetition', 1)}) ---\n")
                f.write(f"Scene type: {self.game_type_desc.get(result['game_type'], result['game_type'])}\n")
                f.write(f"Scene ID: {result['scene_id']}\n")
                f.write(f"Correct scene: {letters[result['correct_index']-1]}\n")
                f.write(f"Predicted scene: {letters[result['predicted_index']-1] if result['predicted_index'] > 0 else 'Invalid prediction'}\n")
                f.write(f"Original correct index: {result.get('original_correct_index', 'N/A')}\n")
                f.write(f"Original predicted index: {result.get('original_predicted_index', 'N/A')}\n")
                f.write(f"Prediction result: {'Correct' if result['correct'] else 'Incorrect'}\n")
                f.write(f"Response time: {result['response_time']:.2f} seconds\n")
                if 'shuffle_mapping' in result:
                    f.write(f"Shuffle mapping: {result['shuffle_mapping']}\n")
                f.write("\nModel response:\n")
                f.write(result['response'])
                f.write("\n" + "="*50 + "\n")

    def extract_predicted_scene_number(self, response_text: str) -> int:
        """
        从模型响应中提取预测的场景编号
        
        Args:
            response_text: 模型的响应文本
            
        Returns:
            预测的场景编号（1-4），如果无法提取则返回-1（表示放弃回答）
        """
        # 打印原始响应文本，用于调试
        self.logger.log("\n=== Model Response Text ===")
        self.logger.log(response_text)
        self.logger.log("==========================")
        
        # 查找Final Result部分
        final_result_match = re.search(r"Final Result:.*?\"I predict that scene ([A-D]) will succeed\.\"", response_text, re.DOTALL)
        if final_result_match:
            letter = final_result_match.group(1)
            self.logger.log(f"\nFound prediction in Final Result: {letter}")
            return {'A': 1, 'B': 2, 'C': 3, 'D': 4}[letter]
        
        # 查找包含字母A-D的关键短语
        patterns = [
            r"I predict scene\s*([A-D])\s*will succeed",
            r"I predict that scene\s*([A-D])\s*will succeed",
            r"Scene\s*([A-D])\s*will succeed",
            r"Choose\s*([A-D])",
            r"Scene\s*([A-D])\s*can succeed",
            r"Select scene\s*([A-D])",
            r"([A-D])\s*will succeed",
            r"Predict\s*([A-D])",
            r"Predict scene\s*([A-D])"
        ]
        
        # 字母到数字的映射
        letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        
        # 记录所有匹配到的字母
        all_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, response_text)
            for match in matches:
                letter = match.group(1)
                if letter in letter_to_number:
                    all_matches.append((letter, pattern, match.group(0)))
        
        # 打印所有匹配结果
        if all_matches:
            self.logger.log("\nMatches found:")
            for letter, pattern, matched_text in all_matches:
                self.logger.log(f"Letter: {letter}, Pattern: {pattern}, Matched text: {matched_text}")
            # 使用第一个匹配的字母并转换为数字
            return letter_to_number[all_matches[0][0]]
        
        # 如果使用上述模式没有找到，尝试查找单独的字母A-D
        letters = re.findall(r'\b[A-D]\b', response_text)
        if letters:
            self.logger.log(f"\nFound standalone letter: {letters[0]}")
            return letter_to_number[letters[0]]
        
        self.logger.log("\nWarning: Model did not make a valid prediction (放弃回答)")
        return -1  # 返回-1表示模型放弃回答
    
    def analyze_error_patterns(self) -> Dict[str, int]:
        """分析错误模式"""
        error_patterns = {
            "Physical Principle Misunderstanding": 0,
            "Visual Feature Misjudgment": 0,
            "Oversimplification": 0,
            "Overlooking Key Details": 0,
            "Other": 0
        }
        
        for result in self.results:
            if not result['correct']:
                response = result['response'].lower()
                
                # 基于响应文本分析错误类型
                if any(word in response for word in ["gravity", "force", "mass", "momentum", "energy"]):
                    error_patterns["Physical Principle Misunderstanding"] += 1
                elif any(word in response for word in ["looks", "appearance", "shape", "position"]):
                    error_patterns["Visual Feature Misjudgment"] += 1
                elif any(word in response for word in ["simple", "basic", "direct"]):
                    error_patterns["Oversimplification"] += 1
                elif any(word in response for word in ["detail", "small", "precise"]):
                    error_patterns["Overlooking Key Details"] += 1
                else:
                    error_patterns["Other"] += 1
        
        return error_patterns

    def find_same_scene_trials(self) -> List[Tuple[List[Path], List[Path]]]:
        """
        查找同一场景下的成功和失败试验
        
        Returns:
            列表，每个元素是一个元组 (失败案例列表, 成功案例列表)
        """
        # 使用字典来组织同一场景的试验
        scene_trials = {}  # 键: (game_type, scene_id), 值: {"success": [], "failure": []}
        
        for subject in self.subjects:
            for trial_folder in subject.iterdir():
                if not trial_folder.is_dir():
                    continue
                
                # 解析文件夹名称
                parts = trial_folder.name.split('_')
                if len(parts) < 5:
                    continue
                
                game_type = parts[0]
                
                # 如果指定了游戏类型，跳过不匹配的类型
                if self.game_type and game_type != self.game_type:
                    continue
                    
                scene_id = parts[3]  # 场景ID，通常是obj1, obj2等
                is_success = parts[-1] == "True"
                
                key = (game_type, scene_id)
                if key not in scene_trials:
                    scene_trials[key] = {"success": [], "failure": []}
                
                if is_success:
                    scene_trials[key]["success"].append(trial_folder)
                else:
                    scene_trials[key]["failure"].append(trial_folder)
        
        # 筛选出同时有至少3个失败和1个成功案例的场景
        valid_scenes = []
        for (game_type, scene_id), trials in scene_trials.items():
            if len(trials["failure"]) >= 3 and len(trials["success"]) >= 1:
                valid_scenes.append((trials["failure"], trials["success"]))
                self.logger.log(f"Found valid scene: {game_type}_{scene_id}, "
                              f"Failure cases: {len(trials['failure'])}, "
                              f"Success cases: {len(trials['success'])}")
        
        return valid_scenes

    def save_results(self, filename: str):
        """
        保存结果到JSON文件
        
        Args:
            filename: 结果文件名
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            # 将结果转换为可序列化的格式
            serializable_results = []
            for result in self.results:
                # 创建结果的副本
                result_copy = result.copy()
                # 移除不可序列化的字段（如Path对象）
                if 'trial_paths' in result_copy:
                    del result_copy['trial_paths']
                serializable_results.append(result_copy)
            
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        self.logger.log(f"Results saved to: {output_path}\n")

    def load_prompt_from_file(self, game_type: str) -> str:
        """
        从文件加载特定游戏类型的提示内容
        
        Args:
            game_type: 游戏类型
            
        Returns:
            提示内容字符串
        """
        # 文件名规则：游戏类型小写.txt
        prompt_file = self.prompt_dir / f"{game_type.lower()}.txt"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            self.logger.log(f"Warning: Prompt file {prompt_file} not found, using default prompt content")
            # 如果找不到提示文件，使用默认内容
            if game_type == "Bridge":
                return 'Please carefully observe the following four bridge building scenes, where the positions of the blue elongated blocks are different in each scene. Your task is to determine which scene has the most suitable position for the elongated blocks to form a stable bridge structure, helping the red ball successfully reach the green target area.\n\nImportant notes:\n1. Scenes are labeled with letters A-D, and the order has no correlation with success probability\n2. Please focus on:\n   - Whether the elongated blocks can form a stable bridge structure\n   - Whether the bridge structure can support the weight of the ball\n   - Whether the position and angle of the bridge are suitable for the ball to pass\n3. In your analysis, consider:\n   - The support points and balance state of the blocks\n   - The stability of the bridge structure\n   - Whether the ball\'s path will be smooth\n   - Whether there are any structural defects\n\nPlease structure your answer as follows:\nReasoning: For each scene, explain step by step what will happen, whether the red ball will reach the green target area, and why you believe scene X has the highest chance of success.\nFinal Result: "I predict that scene X will succeed."'
            else:
                return f'Please carefully observe the following four scenes, where the positions of objects are slightly different in each scene. Your task is to determine which scene will allow the red ball to successfully reach the green target area.\n\nImportant notes:\n1. Scenes are labeled with letters A-D, and the order has no correlation with success probability\n2. Please analyze each scene based on physical principles\n\nPlease structure your answer as follows:\nReasoning: For each scene, explain step by step what will happen, whether the red ball will reach the green target area, and why you believe scene X has the highest chance of success.\nFinal Result: "I predict that scene X will succeed."'

def encode_image_to_base64(image_path):
    """
    Convert image file to base64 string
    
    Args:
        image_path: Path to image file
        
    Returns:
        base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate visual language models\' physical intuition capabilities')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--model_name', type=str, default='gemma3:27b', help='Model name')
    parser.add_argument('--api_key', type=str, default=None, help='API key')
    parser.add_argument('--base_url', type=str, 
                        default="http://localhost:11434/v1", 
                        help='API base URL')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--num_sets', type=int, default=5, help='Number of image sets to test')
    parser.add_argument('--game_type', type=str, default=None, help='Game type to test, e.g. Basic')
    parser.add_argument('--repetitions', type=int, default=4, help='Number of repetitions for each image set (for consistency testing)')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(output_dir)
    
    # 打印API配置信息
    logger.log_section("API Configuration")
    logger.log(f"Model: {args.model_name}")
    logger.log(f"API Base URL: {args.base_url}")
    logger.log(f"API Key provided via command line: {'Yes' if args.api_key else 'No'}")
    logger.log(f"API Key from environment: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    logger.log(f"Max Tokens: {args.max_tokens}")
    logger.log(f"Temperature: {args.temperature}")
    logger.log(f"Top P: {args.top_p}")
    
    evaluator = PhysicalIntuitionEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        output_dir=args.output_dir,
        game_type=args.game_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    evaluator.run_evaluation(num_sets=args.num_sets, repetitions=args.repetitions)
    
    # 记录评估完成时间
    logger.log_section("Evaluation Summary")
    logger.log(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
