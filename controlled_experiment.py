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

class ControlledExperiment:
    def __init__(self, 
                data_root: str, 
                model_name: str = "qwen-vl-plus",
                api_key: Optional[str] = None,
                base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                output_dir: str = "controlled_results"):
        """
        初始化控制变量实验
        
        Args:
            data_root: 数据根目录
            model_name: 要测试的模型名称
            api_key: API密钥
            base_url: API基础URL
            output_dir: 结果输出目录
        """
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
        
        # 获取所有主题文件夹
        self.subjects = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("Subj_")])
        
        # 游戏类型映射到中文描述（用于提示）
        self.game_type_desc = {
            "Basic": "基础物理场景",
            "Bridge": "搭桥场景",
            "Catapult": "弹射器场景",
            "Chaining": "连锁反应场景",
            "Falling": "物体下落场景",
            "Gap": "跨越间隙场景",
            "Launch": "发射场景",
            "Prevention": "阻止运动场景",
            "SeeSaw": "跷跷板场景",
            "Shafts": "轴道场景",
            "Table": "桌面场景",
            "Unbox": "开箱场景",
            "Unsupport": "移除支撑场景"
        }
        
        # 定义实验策略
        self.strategies = {
            # 1-shot策略
            "1shot_success": {"shot_count": 1, "success": [True]},
            "1shot_failure": {"shot_count": 1, "success": [False]},
            
            # 2-shot策略
            "2shot_success": {"shot_count": 2, "success": [True, True]},
            "2shot_failure": {"shot_count": 2, "success": [False, False]},
            "2shot_mixed": {"shot_count": 2, "success": [True, False]},
            
            # 3-shot策略
            "3shot_success": {"shot_count": 3, "success": [True, True, True]},
            "3shot_failure": {"shot_count": 3, "success": [False, False, False]},
            "3shot_mixed_1": {"shot_count": 3, "success": [True, False, False]},
            "3shot_mixed_2": {"shot_count": 3, "success": [True, True, False]},
            
            # 0-shot基线
            "0shot": {"shot_count": 0, "success": []}
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
    
    def prepare_controlled_examples(self, test_trial: Path, strategy_key: str) -> List[Dict]:
        """
        根据特定策略准备示例 - 控制变量版本，采用累积模式
        
        Args:
            test_trial: 要测试的试验
            strategy_key: 策略键名
            
        Returns:
            示例列表
        """
        strategy = self.strategies[strategy_key]
        shot_count = strategy["shot_count"]
        success_pattern = strategy["success"]
        
        if shot_count == 0:
            return []
        
        # 获取测试试验的游戏类型和工具类型
        test_parts = test_trial.name.split('_')
        test_game_type = test_parts[0]
        test_tool_type = test_parts[3] if len(test_parts) >= 4 else None
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验
        all_trials = [t for t in all_trials if t != test_trial]
        
        examples = []
        
        # 确定基础策略
        base_strategy = None
        
        # 2-shot基于1-shot构建
        if shot_count == 2:
            if strategy_key == "2shot_success":
                # 成功+成功，基于1-shot失败
                base_strategy = "1shot_success"
            elif strategy_key == "2shot_failure":
                # 失败+失败，基于1-shot成功
                base_strategy = "1shot_failure"
            elif strategy_key == "2shot_mixed":
                # 根据第一个元素确定基础
                if success_pattern[0]:
                    base_strategy = "1shot_failure"  # 基于1-shot失败
                else:
                    base_strategy = "1shot_success"  # 基于1-shot成功
        
        # 3-shot基于2-shot构建
        elif shot_count == 3:
            if strategy_key == "3shot_success":
                base_strategy = "2shot_success"  # 基于混合策略
            elif strategy_key == "3shot_failure":
                base_strategy = "2shot_failure"  # 基于混合策略
            elif strategy_key == "3shot_mixed_1":
                base_strategy = "2shot_success"  # 基于两个成功
            elif strategy_key == "3shot_mixed_2":
                base_strategy = "2shot_failure"  # 基于两个失败
        
        # 如果有基础策略，先获取基础示例
        if base_strategy:
            base_examples = self.prepare_controlled_examples_base(test_trial, base_strategy)
            if base_examples:
                examples.extend(base_examples)
        
        # 计算还需要添加多少示例
        remaining_count = shot_count - len(examples)
        
        # 如果基础示例不足或没有基础策略，使用原方法选择剩余示例
        if remaining_count > 0:
            # 确定剩余需要的示例模式
            remaining_pattern = success_pattern[len(examples):]
            
            for needs_success in remaining_pattern:
                # 首先尝试找到相同游戏类型+相同工具类型的案例
                same_type_tool_candidates = [
                    t for t in all_trials 
                    if (t.name.split('_')[0] == test_game_type) and 
                       (len(t.name.split('_')) >= 4 and t.name.split('_')[3] == test_tool_type) and
                       (t.name.endswith("True") == needs_success)
                ]
                
                # 如果找不到足够的案例，尝试仅相同游戏类型的案例
                if not same_type_tool_candidates:
                    same_type_candidates = [
                        t for t in all_trials 
                        if (t.name.split('_')[0] == test_game_type) and
                           (t.name.endswith("True") == needs_success)
                    ]
                    
                    # 如果仍找不到，使用任何符合成功/失败要求的案例
                    if not same_type_candidates:
                        any_candidates = [
                            t for t in all_trials
                            if t.name.endswith("True") == needs_success
                        ]
                        
                        if not any_candidates:
                            print(f"警告: 找不到{'成功' if needs_success else '失败'}案例，跳过")
                            return examples  # 返回已有的示例，可能不满足要求
                        
                        candidates = any_candidates
                    else:
                        candidates = same_type_candidates
                else:
                    candidates = same_type_tool_candidates
                
                # 已经使用过的试验不再使用，防止重复
                used_trials = [e["trial_path"] for e in examples]
                unused_candidates = [c for c in candidates if c not in used_trials]
                if unused_candidates:
                    candidates = unused_candidates
                
                # 随机选择一个案例
                selected_trial = random.choice(candidates)
                
                # 检查是否有完整的帧
                try:
                    first_frame = next(selected_trial.glob("frame_0000.png"))
                    last_frame = max(selected_trial.glob("frame_*.png"), key=lambda p: int(p.stem.split('_')[1]))
                    
                    game_type = selected_trial.name.split('_')[0]
                    is_success = selected_trial.name.endswith("True")
                    
                    examples.append({
                        "trial_path": selected_trial,
                        "initial_frame": str(first_frame),
                        "final_frame": str(last_frame),
                        "is_success": is_success,
                        "game_type": game_type,
                        "description": f"在这个{self.game_type_desc.get(game_type, game_type)}中，红球{'成功' if is_success else '未能'}到达绿色目标。"
                    })
                except (StopIteration, ValueError):
                    print(f"警告: {selected_trial} 中帧不完整，跳过")
                    continue
        
        return examples
    
    def prepare_controlled_examples_base(self, test_trial: Path, base_strategy: str) -> List[Dict]:
        """
        获取基础策略的示例，用于构建累积模式的示例
        
        Args:
            test_trial: 要测试的试验
            base_strategy: 基础策略名称
            
        Returns:
            基础示例列表
        """
        # 避免递归过深，直接获取基础示例而不再调用累积模式
        strategy = self.strategies[base_strategy]
        shot_count = strategy["shot_count"]
        success_pattern = strategy["success"]
        
        # 获取测试试验的游戏类型和工具类型
        test_parts = test_trial.name.split('_')
        test_game_type = test_parts[0]
        test_tool_type = test_parts[3] if len(test_parts) >= 4 else None
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验
        all_trials = [t for t in all_trials if t != test_trial]
        
        examples = []
        
        # 按照success_pattern选择示例
        for needs_success in success_pattern:
            # 选择逻辑与原方法相同
            same_type_tool_candidates = [
                t for t in all_trials 
                if (t.name.split('_')[0] == test_game_type) and 
                   (len(t.name.split('_')) >= 4 and t.name.split('_')[3] == test_tool_type) and
                   (t.name.endswith("True") == needs_success)
            ]
            
            if not same_type_tool_candidates:
                same_type_candidates = [
                    t for t in all_trials 
                    if (t.name.split('_')[0] == test_game_type) and
                       (t.name.endswith("True") == needs_success)
                ]
                
                if not same_type_candidates:
                    any_candidates = [
                        t for t in all_trials
                        if t.name.endswith("True") == needs_success
                    ]
                    
                    if not any_candidates:
                        continue
                    
                    candidates = any_candidates
                else:
                    candidates = same_type_candidates
            else:
                candidates = same_type_tool_candidates
            
            # 防止重复
            used_trials = [e["trial_path"] for e in examples]
            unused_candidates = [c for c in candidates if c not in used_trials]
            if unused_candidates:
                candidates = unused_candidates
            
            # 选择示例
            selected_trial = random.choice(candidates)
            
            try:
                first_frame = next(selected_trial.glob("frame_0000.png"))
                last_frame = max(selected_trial.glob("frame_*.png"), key=lambda p: int(p.stem.split('_')[1]))
                
                game_type = selected_trial.name.split('_')[0]
                is_success = selected_trial.name.endswith("True")
                
                examples.append({
                    "trial_path": selected_trial,
                    "initial_frame": str(first_frame),
                    "final_frame": str(last_frame),
                    "is_success": is_success,
                    "game_type": game_type,
                    "description": f"在这个{self.game_type_desc.get(game_type, game_type)}中，红球{'成功' if is_success else '未能'}到达绿色目标。"
                })
            except (StopIteration, ValueError):
                continue
        
        return examples
    
    def build_prompt(self, test_trial: Path, examples: List[Dict]) -> Tuple[List, bool, str]:
        """
        构建提示
        
        Args:
            test_trial: 要测试的试验
            examples: 示例列表
            
        Returns:
            消息列表、真实结果和游戏类型
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
        
        messages = []
        
        # 系统提示
        system_message = {
            "role": "system", 
            "content": "你是一个具备强大物理直觉的AI助手。你需要根据物理场景图像，预测红色球体是否能够到达绿色目标区域。"
        }
        messages.append(system_message)
        
        # 添加示例
        for i, example in enumerate(examples):
            base64_initial = encode_image_to_base64(example['initial_frame'])
            base64_final = encode_image_to_base64(example['final_frame'])
            
            example_content = [
                {"type": "text", "text": f"示例{i+1}的初始状态:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_initial}"}},
                {"type": "text", "text": f"示例{i+1}的最终状态:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_final}"}},
                {"type": "text", "text": example["description"]}
            ]
            messages.append({"role": "user", "content": example_content})
            messages.append({"role": "assistant", "content": f"我理解了，在这个{self.game_type_desc.get(example['game_type'], example['game_type'])}中，红球{'成功' if example['is_success'] else '未能'}到达绿色目标区域。"})
        
        # 添加测试问题
        shot_count = len(examples)
        question_text = "根据初始场景图像，判断红色球体能否最终到达绿色目标区域？请详细解释你的推理过程，包括可能影响球体运动的物理因素。你的回答应该包含明确的是/否预测。"
        
        if shot_count > 0:
            question_text = f"请基于前面的{shot_count}个示例和你的物理知识，" + question_text
        
        base64_first = encode_image_to_base64(str(first_frame))
        question_content = [
            {"type": "text", "text": question_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_first}"}}
        ]
        
        messages.append({"role": "user", "content": question_content})
        
        return messages, true_result, game_type
    
    def call_model(self, messages: List[Dict]) -> Dict:
        """
        调用模型
        
        Args:
            messages: 消息列表
            
        Returns:
            模型响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response
        except Exception as e:
            print(f"API调用错误: {e}")
            # 等待一下再重试
            time.sleep(3)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                return response
            except Exception as e2:
                print(f"重试失败: {e2}")
                return None
    
    def parse_prediction(self, response_text: str) -> Tuple[bool, str, float]:
        """
        从响应文本中解析预测结果并评估推理质量
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            预测结果（True表示成功，False表示失败）、推理类型和推理质量分数
        """
        # 简单的关键词匹配（可以根据实际响应进行改进）
        positive_keywords = ["能够", "可以", "会", "能", "是的", "是", "成功"]
        negative_keywords = ["不能", "不可以", "不会", "不", "否", "否的", "失败"]
        
        # 评估推理质量
        reasoning_type, reasoning_score = self.evaluate_reasoning(response_text)
        
        # 检查否定词后是否跟着肯定词（例如"不能不"这种双重否定）
        for neg in negative_keywords:
            for pos in positive_keywords:
                if f"{neg}{pos}" in response_text:
                    return True, reasoning_type, reasoning_score  # 双重否定视为肯定
        
        # 检查肯定和否定关键词
        for neg in negative_keywords:
            if neg in response_text:
                return False, reasoning_type, reasoning_score
        
        for pos in positive_keywords:
            if pos in response_text:
                return True, reasoning_type, reasoning_score
        
        # 如果无法确定，默认为否定
        return False, reasoning_type, reasoning_score
    
    def evaluate_reasoning(self, response_text: str) -> Tuple[str, float]:
        """评估模型推理过程的质量"""
        # 物理关键概念
        physics_concepts = ["重力", "动量", "能量", "惯性", "摩擦", "加速度", "力", ...]
        
        # 因果推理标记
        causal_markers = ["因为", "所以", "导致", "引起", "造成", ...]
        
        # 推理步骤标记
        reasoning_markers = ["首先", "然后", "接下来", "最后", "因此", ...]
        
        # 随机猜测标记
        guessing_markers = ["猜测", "可能", "或许", "也许", "大概", ...]
        
        # 计算各类标记的出现次数
        physics_count = sum(1 for concept in physics_concepts if concept in response_text)
        causal_count = sum(1 for marker in causal_markers if marker in response_text)
        reasoning_count = sum(1 for marker in reasoning_markers if marker in response_text)
        guessing_count = sum(1 for marker in guessing_markers if marker in response_text)
        
        # 计算推理质量分数
        quality_score = (物理概念权重 * 物理概念数量 + 
                        因果推理权重 * 因果推理标记数量 + 
                        推理步骤权重 * 推理步骤标记数量 - 
                        猜测标记权重 * 猜测标记数量 +
                        响应长度因子)
        
        # 确定推理类型
        if quality_score < 0.3:
            reasoning_type = "随机猜测"
        elif quality_score < 0.6:
            reasoning_type = "简单推理"
        else:
            reasoning_type = "深入推理"
        
        return reasoning_type, quality_score
    
    def run_experiment(self, num_trials: int = 10, game_types: List[str] = None):
        """
        运行控制变量实验，使用累积模式，确保示例完整性和推理过程分析
        
        Args:
            num_trials: 每个策略的试验数量
            game_types: 要测试的游戏类型列表
        """
        # 重置结果
        self.results = []
        
        # 选择多样化的试验样本
        test_trials = self.select_diverse_trials(num_trials, game_types)
        
        # 创建一个字典来存储每个测试样本的示例累积链
        # 格式为: {trial_path: {strategy_name: [示例列表]}}
        sample_examples = {str(trial): {} for trial in test_trials}
        
        # 排序策略以确保先处理0-shot，然后低shot数到高shot数
        strategies_by_shot = sorted(self.strategies.items(), key=lambda x: x[1]["shot_count"])
        
        # 先处理0-shot策略，建立基线对照
        for strategy_name, strategy_info in strategies_by_shot:
            if strategy_info["shot_count"] != 0:
                continue
            
            print(f"\n===== 策略: {strategy_name} =====")
            
            for trial in test_trials:
                print(f"测试: {trial.name}")
                
                # 0-shot没有示例
                examples = []
                
                # 构建提示
                messages, true_result, game_type = self.build_prompt(trial, examples)
                
                if messages is None:
                    print(f"  警告: 跳过 {trial.name}")
                    continue
                
                # 调用模型
                response = self.call_model(messages)
                if not response:
                    print(f"  错误: 模型调用失败")
                    continue
                
                # 解析预测
                response_text = response.choices[0].message.content
                prediction, reasoning_type, reasoning_score = self.parse_prediction(response_text)
                
                # 记录结果
                result = {
                    "trial": trial.name,
                    "game_type": game_type,
                    "strategy": strategy_name,
                    "shot_count": 0,
                    "true_result": true_result,
                    "prediction": prediction,
                    "correct": prediction == true_result,
                    "response": response_text,
                    "examples": [],  # 0-shot没有示例
                    "reasoning_type": reasoning_type,
                    "reasoning_score": reasoning_score
                }
                
                self.results.append(result)
                print(f"  结果: {result['correct']} (推理: {reasoning_type}, 分数: {reasoning_score:.2f})")
                
                # 保存0-shot的结果，可以供后续策略参考
                sample_examples[str(trial)][strategy_name] = []
            
            # 保存中间结果
            self.save_results("controlled_results_intermediate.json")
        
        # 然后处理所有1-shot策略，为每个测试样本建立基础示例
        for strategy_name, strategy_info in strategies_by_shot:
            shot_count = strategy_info["shot_count"]
            
            # 跳过非1-shot策略
            if shot_count != 1:
                continue
            
            print(f"\n===== 策略: {strategy_name} =====")
            
            for trial in test_trials:
                print(f"测试: {trial.name}")
                
                # 准备特定策略的示例
                examples = self.prepare_controlled_examples(trial, strategy_name)
                
                # 确保示例完整性
                examples = self.ensure_examples_consistency(strategy_name, trial, examples)
                
                if len(examples) == shot_count:
                    # 存储这个样本的1-shot示例，以便后续策略使用
                    sample_examples[str(trial)][strategy_name] = examples
                    
                    # 构建提示
                    messages, true_result, game_type = self.build_prompt(trial, examples)
                    
                    if messages is None:
                        print(f"  警告: 跳过 {trial.name}")
                        continue
                    
                    # 调用模型
                    response = self.call_model(messages)
                    if not response:
                        print(f"  错误: 模型调用失败")
                        continue
                    
                    # 解析预测
                    response_text = response.choices[0].message.content
                    prediction, reasoning_type, reasoning_score = self.parse_prediction(response_text)
                    
                    # 记录结果
                    result = {
                        "trial": trial.name,
                        "game_type": game_type,
                        "strategy": strategy_name,
                        "shot_count": shot_count,
                        "true_result": true_result,
                        "prediction": prediction,
                        "correct": prediction == true_result,
                        "response": response_text,
                        "examples": [ex["description"] for ex in examples],
                        "base_strategy": "0shot",  # 记录基础策略为0-shot
                        "reasoning_type": reasoning_type,
                        "reasoning_score": reasoning_score
                    }
                    
                    self.results.append(result)
                    print(f"  结果: {result['correct']} (推理: {reasoning_type}, 分数: {reasoning_score:.2f})")
                else:
                    print(f"  警告: 无法为策略 {strategy_name} 准备足够示例，跳过")
                
                # 保存中间结果
                self.save_results("controlled_results_intermediate.json")
        
        # 然后处理2-shot和3-shot策略，基于已有的示例链
        for strategy_name, strategy_info in strategies_by_shot:
            shot_count = strategy_info["shot_count"]
            
            # 跳过0-shot和1-shot策略
            if shot_count <= 1:
                continue
            
            print(f"\n===== 策略: {strategy_name} =====")
            
            for trial in test_trials:
                print(f"测试: {trial.name}")
                
                # 确定这个策略的基础策略
                base_strategy = self.determine_base_strategy(strategy_name, trial, sample_examples)
                
                if base_strategy and str(trial) in sample_examples and base_strategy in sample_examples[str(trial)]:
                    # 获取基础示例
                    base_examples = sample_examples[str(trial)][base_strategy]
                    
                    # 在基础示例上添加新的示例
                    examples = self.extend_examples(trial, strategy_name, base_examples)
                else:
                    # 如果找不到基础示例，使用常规方法准备示例
                    examples = self.prepare_controlled_examples(trial, strategy_name)
                
                # 确保示例完整性
                examples = self.ensure_examples_consistency(strategy_name, trial, examples)
                
                if len(examples) == shot_count:
                    # 存储这个样本的示例，以便后续策略使用
                    sample_examples[str(trial)][strategy_name] = examples
                    
                    # 构建提示
                    messages, true_result, game_type = self.build_prompt(trial, examples)
                    
                    if messages is None:
                        print(f"  警告: 跳过 {trial.name}")
                        continue
                    
                    # 调用模型
                    response = self.call_model(messages)
                    if not response:
                        print(f"  错误: 模型调用失败")
                        continue
                    
                    # 解析预测
                    response_text = response.choices[0].message.content
                    prediction, reasoning_type, reasoning_score = self.parse_prediction(response_text)
                    
                    # 记录结果
                    result = {
                        "trial": trial.name,
                        "game_type": game_type,
                        "strategy": strategy_name,
                        "shot_count": shot_count,
                        "true_result": true_result,
                        "prediction": prediction,
                        "correct": prediction == true_result,
                        "response": response_text,
                        "examples": [ex["description"] for ex in examples],
                        "base_strategy": base_strategy,
                        "reasoning_type": reasoning_type,
                        "reasoning_score": reasoning_score
                    }
                    
                    self.results.append(result)
                    print(f"  结果: {result['correct']} (推理: {reasoning_type}, 分数: {reasoning_score:.2f})")
                else:
                    print(f"  警告: 无法为策略 {strategy_name} 准备足够示例，跳过")
                
                # 保存中间结果
                self.save_results("controlled_results_intermediate.json")
        
        # 保存最终结果
        self.save_results("controlled_results_final.json")
        
        # 分析结果
        self.analyze_results()
    
    def select_diverse_trials(self, num_trials=10, game_types=None):
        """
        选择多样化的试验
        
        Args:
            num_trials: 试验数量
            game_types: 游戏类型列表
            
        Returns:
            试验列表
        """
        all_trials = self.find_trial_folders(game_types=game_types)
        
        if not all_trials:
            raise ValueError("找不到符合条件的试验")
        
        # 按游戏类型分组
        game_type_trials = {}
        for trial in all_trials:
            game_type = trial.name.split('_')[0]
            if game_type not in game_type_trials:
                game_type_trials[game_type] = []
            game_type_trials[game_type].append(trial)
        
        # 确保成功和失败案例的平衡
        selected_trials = []
        available_types = list(game_type_trials.keys())
        
        while len(selected_trials) < num_trials and available_types:
            # 从可用类型中随机选择一个
            game_type = random.choice(available_types)
            
            # 对于每种类型，尝试选择一个成功和一个失败案例
            success_trials = [t for t in game_type_trials[game_type] if t.name.endswith("True")]
            failure_trials = [t for t in game_type_trials[game_type] if t.name.endswith("False")]
            
            if success_trials:
                selected_trials.append(random.choice(success_trials))
                if len(selected_trials) >= num_trials:
                    break
            
            if failure_trials:
                selected_trials.append(random.choice(failure_trials))
                if len(selected_trials) >= num_trials:
                    break
            
            # 从可用类型中移除已处理的类型
            available_types.remove(game_type)
        
        # 如果仍需要更多样本，随机选择
        if len(selected_trials) < num_trials:
            remaining = num_trials - len(selected_trials)
            remaining_trials = [t for t in all_trials if t not in selected_trials]
            
            if remaining_trials:
                selected_trials.extend(random.sample(remaining_trials, min(remaining, len(remaining_trials))))
        
        return selected_trials[:num_trials]
    
    def save_results(self, filename: str):
        """保存结果到JSON文件"""
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def analyze_results(self):
        """分析结果，增加推理质量分析"""
        if not self.results:
            print("没有结果可分析")
            return
        
        df = pd.DataFrame(self.results)
        
        # 按策略分组计算准确率
        strategy_accuracy = df.groupby('strategy')['correct'].mean()
        strategy_counts = df.groupby('strategy').size()
        
        # 按shot数量分组计算准确率
        shot_count_accuracy = df.groupby('shot_count')['correct'].mean()
        
        # 按推理类型分组计算准确率
        reasoning_type_accuracy = df.groupby('reasoning_type')['correct'].mean()
        reasoning_type_counts = df.groupby('reasoning_type').size()
        
        # 计算推理分数与准确率的相关性
        correlation = df['reasoning_score'].corr(df['correct'].astype(float))
        
        # 按游戏类型和策略分组计算准确率
        if len(df['game_type'].unique()) > 1:
            game_strategy_accuracy = df.groupby(['game_type', 'strategy'])['correct'].mean().unstack()
        
        print("\n===== 控制变量实验结果 =====")
        print(f"总样本数: {len(df)}")
        print(f"总体准确率: {df['correct'].mean():.4f}")
        
        print("\n按策略的准确率:")
        for strategy in strategy_accuracy.index:
            print(f"{strategy}: {strategy_accuracy[strategy]:.4f} (n={strategy_counts[strategy]})")
        
        print("\n按Shot数量的准确率:")
        print(shot_count_accuracy)
        
        print("\n按推理类型的准确率:")
        for r_type in reasoning_type_accuracy.index:
            print(f"{r_type}: {reasoning_type_accuracy[r_type]:.4f} (n={reasoning_type_counts[r_type]})")
        
        print(f"\n推理质量分数与准确率的相关性: {correlation:.4f}")
        
        # 绘制策略准确率对比图
        plt.figure(figsize=(14, 8))
        bars = strategy_accuracy.plot(kind='bar', color='skyblue')
        plt.title('不同策略的准确率对比')
        plt.xlabel('策略')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(strategy_accuracy):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 按shot数量对策略进行分组着色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        shot_groups = {'0shot': 0, 
                      '1shot_success': 1, '1shot_failure': 1,
                      '2shot_success': 2, '2shot_failure': 2, '2shot_mixed': 2,
                      '3shot_success': 3, '3shot_failure': 3, '3shot_mixed_1': 3, '3shot_mixed_2': 3}
        
        for i, bar in enumerate(bars.patches):
            strategy = strategy_accuracy.index[i]
            shot_group = shot_groups.get(strategy, 0)
            bar.set_color(colors[shot_group % len(colors)])
        
        plt.savefig(self.output_dir / 'strategy_comparison.png')
        
        # 绘制shot数量准确率对比图
        plt.figure(figsize=(10, 6))
        shot_count_accuracy.plot(kind='bar', color='lightgreen')
        plt.title('不同Shot数量的准确率对比')
        plt.xlabel('Shot数量')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(shot_count_accuracy):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
        plt.savefig(self.output_dir / 'shot_count_comparison.png')
        
        # 绘制推理类型准确率对比图
        plt.figure(figsize=(10, 6))
        reasoning_type_accuracy.plot(kind='bar', color='salmon')
        plt.title('不同推理类型的准确率对比')
        plt.xlabel('推理类型')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(reasoning_type_accuracy):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(self.output_dir / 'reasoning_type_comparison.png')
        
        # 绘制推理质量分数与准确率的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(df['reasoning_score'], df['correct'].astype(float), alpha=0.5)
        plt.title('推理质量分数与准确率的关系')
        plt.xlabel('推理质量分数')
        plt.ylabel('是否正确')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(self.output_dir / 'reasoning_quality_correlation.png')
        
        # 保存统计数据
        with open(self.output_dir / 'controlled_statistics.txt', 'w') as f:
            f.write(f"总样本数: {len(df)}\n")
            f.write(f"总体准确率: {df['correct'].mean():.4f}\n\n")
            
            f.write("按策略的准确率:\n")
            for strategy in strategy_accuracy.index:
                f.write(f"{strategy}: {strategy_accuracy[strategy]:.4f} (n={strategy_counts[strategy]})\n")
            
            f.write("\n按Shot数量的准确率:\n")
            f.write(str(shot_count_accuracy) + "\n\n")
            
            f.write("\n按推理类型的准确率:\n")
            for r_type in reasoning_type_accuracy.index:
                f.write(f"{r_type}: {reasoning_type_accuracy[r_type]:.4f} (n={reasoning_type_counts[r_type]})\n")
            
            f.write(f"\n推理质量分数与准确率的相关性: {correlation:.4f}\n\n")
            
            f.write("按策略的推理类型分布:\n")
            f.write(str(df.groupby(['strategy', 'reasoning_type']).size().unstack()) + "\n\n")
            
            f.write("按策略的成功/失败预测分布:\n")
            f.write(str(df.groupby(['strategy', 'prediction']).size().unstack()) + "\n\n")
            
            f.write("按策略的真实结果分布:\n")
            f.write(str(df.groupby(['strategy', 'true_result']).size().unstack()) + "\n\n")
            
            if len(df['game_type'].unique()) > 1:
                f.write("按游戏类型和策略的准确率:\n")
                if 'game_strategy_accuracy' in locals():
                    f.write(str(game_strategy_accuracy))
    
    def determine_base_strategy(self, strategy_name: str, trial: Path, sample_examples: Dict) -> Optional[str]:
        """
        确定基础策略，即当前策略应该基于哪个策略构建
        
        Args:
            strategy_name: 当前策略名称
            trial: 测试样本
            sample_examples: 每个样本的示例累积链
            
        Returns:
            基础策略名称或None
        """
        shot_count = self.strategies[strategy_name]["shot_count"]
        success_pattern = self.strategies[strategy_name]["success"]
        
        # 2-shot策略基于1-shot构建
        if shot_count == 2:
            # 当前策略的第一个示例是什么类型（成功/失败）
            first_pattern = success_pattern[0]
            
            # 根据第一个需要的示例类型，确定对应的1-shot策略
            if first_pattern:  # 需要成功案例
                base_name = "1shot_success"
            else:  # 需要失败案例
                base_name = "1shot_failure"
            
            # 检查该样本是否有这个基础策略的示例
            if str(trial) in sample_examples and base_name in sample_examples[str(trial)]:
                return base_name
        
        # 3-shot策略基于2-shot构建
        elif shot_count == 3:
            # 尝试查找合适的2-shot策略
            for base_name in ["2shot_success", "2shot_mixed", "2shot_failure"]:
                if str(trial) in sample_examples and base_name in sample_examples[str(trial)]:
                    base_examples = sample_examples[str(trial)][base_name]
                    
                    # 检查前两个示例的模式是否与当前策略匹配
                    if len(base_examples) == 2:
                        base_pattern = [ex["is_success"] for ex in base_examples]
                        if base_pattern == success_pattern[:2]:
                            return base_name
        
        return None
    
    def extend_examples(self, test_trial: Path, strategy_name: str, base_examples: List[Dict]) -> List[Dict]:
        """
        在基础示例的基础上扩展示例，添加额外的示例
        
        Args:
            test_trial: 测试样本
            strategy_name: 当前策略名称
            base_examples: 基础示例列表
            
        Returns:
            扩展后的示例列表
        """
        # 复制基础示例
        examples = base_examples.copy()
        
        # 确定需要添加的示例数量和类型
        strategy = self.strategies[strategy_name]
        shot_count = strategy["shot_count"]
        success_pattern = strategy["success"]
        
        # 计算还需要添加多少示例
        remaining_count = shot_count - len(examples)
        
        if remaining_count <= 0:
            return examples
        
        # 确定剩余需要的示例模式
        remaining_pattern = success_pattern[len(examples):]
        
        # 获取测试试验的游戏类型和工具类型
        test_parts = test_trial.name.split('_')
        test_game_type = test_parts[0]
        test_tool_type = test_parts[3] if len(test_parts) >= 4 else None
        
        # 获取所有试验
        all_trials = self.find_trial_folders()
        
        # 排除当前测试试验和已使用的示例
        used_trials = [e["trial_path"] for e in examples]
        all_trials = [t for t in all_trials if t != test_trial and t not in used_trials]
        
        # 按照剩余模式添加示例
        for needs_success in remaining_pattern:
            # 首先尝试找到相同游戏类型+相同工具类型的案例
            same_type_tool_candidates = [
                t for t in all_trials 
                if (t.name.split('_')[0] == test_game_type) and 
                    (len(t.name.split('_')) >= 4 and t.name.split('_')[3] == test_tool_type) and
                    (t.name.endswith("True") == needs_success)
            ]
            
            # 如果找不到足够的案例，尝试仅相同游戏类型的案例
            if not same_type_tool_candidates:
                same_type_candidates = [
                    t for t in all_trials 
                    if (t.name.split('_')[0] == test_game_type) and
                        (t.name.endswith("True") == needs_success)
                ]
                
                # 如果仍找不到，使用任何符合成功/失败要求的案例
                if not same_type_candidates:
                    any_candidates = [
                        t for t in all_trials
                        if t.name.endswith("True") == needs_success
                    ]
                    
                    if not any_candidates:
                        print(f"警告: 找不到{'成功' if needs_success else '失败'}案例，跳过")
                        return examples  # 返回已有的示例，可能不满足要求
                    
                    candidates = any_candidates
                else:
                    candidates = same_type_candidates
            else:
                candidates = same_type_tool_candidates
            
            # 已经使用过的试验不再使用，防止重复
            used_trials = [e["trial_path"] for e in examples]
            unused_candidates = [c for c in candidates if c not in used_trials]
            if unused_candidates:
                candidates = unused_candidates
            
            # 随机选择一个案例
            selected_trial = random.choice(candidates)
            
            # 检查是否有完整的帧
            try:
                first_frame = next(selected_trial.glob("frame_0000.png"))
                last_frame = max(selected_trial.glob("frame_*.png"), key=lambda p: int(p.stem.split('_')[1]))
                
                game_type = selected_trial.name.split('_')[0]
                is_success = selected_trial.name.endswith("True")
                
                examples.append({
                    "trial_path": selected_trial,
                    "initial_frame": str(first_frame),
                    "final_frame": str(last_frame),
                    "is_success": is_success,
                    "game_type": game_type,
                    "description": f"在这个{self.game_type_desc.get(game_type, game_type)}中，红球{'成功' if is_success else '未能'}到达绿色目标。"
                })
            except (StopIteration, ValueError):
                print(f"警告: {selected_trial} 中帧不完整，跳过")
                continue
            
            # 更新已使用试验列表
            all_trials = [t for t in all_trials if t != selected_trial]
        
        return examples
    
    def ensure_examples_consistency(self, strategy_name: str, trial: Path, examples: List[Dict]) -> List[Dict]:
        """确保示例的完整性和一致性，如果示例不足，则尝试补全"""
        strategy = self.strategies[strategy_name]
        shot_count = strategy["shot_count"]
        
        # 如果示例不足，尝试补全
        if len(examples) < shot_count:
            print(f"  警告: 策略 {strategy_name} 的示例数量不足，尝试补全")
            
            # 尝试使用常规方法准备补充示例
            additional_examples = self.prepare_controlled_examples(trial, strategy_name)
            
            # 过滤掉已有的示例，避免重复
            used_trials = [e["trial_path"] for e in examples]
            additional_examples = [e for e in additional_examples if e["trial_path"] not in used_trials]
            
            # 添加额外示例
            examples.extend(additional_examples[:needed])
        
        return examples


def encode_image_to_base64(image_path):
    """
    将图像文件编码为base64字符串
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        base64编码的图像字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='控制变量的物理直觉实验')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--model_name', type=str, default='qwen-vl-plus', help='模型名称')
    parser.add_argument('--api_key', type=str, default=None, help='API密钥')
    parser.add_argument('--base_url', type=str, 
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                        help='API基础URL')
    parser.add_argument('--output_dir', type=str, default='controlled_results', help='结果输出目录')
    parser.add_argument('--num_trials', type=int, default=10, help='每个策略的试验数量')
    parser.add_argument('--game_types', type=str, nargs='+', default=None, help='要测试的游戏类型列表')
    
    args = parser.parse_args()
    
    experiment = ControlledExperiment(
        data_root=args.data_root,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    experiment.run_experiment(
        num_trials=args.num_trials,
        game_types=args.game_types
    )


if __name__ == "__main__":
    main() 