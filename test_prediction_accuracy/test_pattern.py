#!/usr/bin/env python3
import os
import sys
import argparse
from test_ollama import PhysicalIntuitionOllamaEvaluator
from pathlib import Path
import random
import json
import time
from typing import List, Dict, Tuple, Optional

class PatternBasedEvaluator(PhysicalIntuitionOllamaEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_results = {}
        self.used_examples = {}  # 记录每个游戏类型已使用的示例
        
    def initialize_game_type(self, game_type: str):
        """初始化特定游戏类型的实验"""
        if game_type not in self.used_examples:
            self.used_examples[game_type] = set()
            
    def find_examples_by_pattern(self, game_type: str, pattern: str, num_examples: int = None) -> List[Path]:
        """Find example trials based on pattern requirements"""
        # 确保初始化
        self.initialize_game_type(game_type)
        
        # 获取所有可用的trials（排除已使用的示例）
        all_trials = set(self.find_trial_folders(game_types=[game_type]))
        available_trials = list(all_trials - self.used_examples[game_type])
        
        # 分离成功和失败案例
        success_trials = [t for t in available_trials if t.name.endswith("True")]
        failure_trials = [t for t in available_trials if t.name.endswith("False")]
        
        selected_examples = []
        
        if pattern == "no_examples":
            return []
        elif pattern == "success_only" and success_trials:
            selected_examples = random.sample(success_trials, 1)
        elif pattern == "failure_only" and failure_trials:
            selected_examples = random.sample(failure_trials, 1)
        elif pattern == "all_success" and len(success_trials) >= 2:
            selected_examples = random.sample(success_trials, 2)
        elif pattern == "all_failure" and len(failure_trials) >= 2:
            selected_examples = random.sample(failure_trials, 2)
        elif pattern == "best_plus_success":
            if num_examples and len(success_trials) >= 1:
                selected_examples = random.sample(success_trials, 1)
        elif pattern == "best_plus_failure":
            if num_examples and len(failure_trials) >= 1:
                selected_examples = random.sample(failure_trials, 1)
        
        # 记录使用的示例
        self.used_examples[game_type].update(selected_examples)
        
        return selected_examples
    
    def build_prompt_with_examples(self, test_trial: Path, examples: List[Path]) -> Tuple[List, bool, str]:
        """Build prompt with example trials"""
        messages, true_result, game_type = self.build_prompt(test_trial)
        if messages is None:
            return None, None, None
            
        # Add examples before the test question
        for example in examples:
            example_messages, example_result, _ = self.build_prompt(example)
            if example_messages is None:
                continue
                
            # Add example Q&A pair
            messages.extend([
                {"role": "user", "content": example_messages[1]["content"]},
                {"role": "assistant", "content": f"{'YES' if example_result else 'NO'}. This is an example where the red ball {'will' if example_result else 'will not'} reach the green target area."}
            ])
            
        return messages, true_result, game_type
    
    def run_pattern_experiment(self, game_type: str, pattern: str, num_trials: int = 10, additional_examples: List[Path] = None):
        """Run experiment for a specific pattern"""
        print(f"\n===== Running {pattern} experiment for {game_type} =====")
        
        # 获取示例
        examples = self.find_examples_by_pattern(game_type, pattern)
        if additional_examples:
            examples.extend(additional_examples)
        
        # 获取测试trials（排除所有已用作示例的trials）
        all_trials = set(self.find_trial_folders(game_types=[game_type]))
        available_trials = list(all_trials - self.used_examples[game_type])
        
        if len(available_trials) > num_trials:
            test_trials = random.sample(available_trials, num_trials)
        else:
            test_trials = available_trials
        
        results = []
        for trial in test_trials:
            messages, true_result, _ = self.build_prompt_with_examples(trial, examples)
            if messages is None:
                continue
                
            response = self.call_model(messages)
            if not response:
                continue
                
            response_text = response.choices[0].message.content
            prediction = self.parse_prediction(response_text)
            
            result = {
                "trial": trial.name,
                "true_result": true_result,
                "prediction": prediction,
                "correct": prediction == true_result,
                "response": response_text
            }
            results.append(result)
            
            # Short pause to avoid API limits
            time.sleep(1)
            
        return results
    
    def run_all_patterns(self, num_trials: int = 10):
        """Run experiments for all patterns and game types"""
        patterns = {
            "0-shot": ["no_examples"],
            "1-shot": ["success_only", "failure_only"],
            "2-shot": ["all_success", "all_failure"],
        }
        
        # 获取所有游戏类型
        game_types = set()
        for trial in self.find_trial_folders():
            game_type = trial.name.split('_')[0]
            game_types.add(game_type)
        
        for game_type in sorted(game_types):
            print(f"\n=== Starting experiments for {game_type} ===")
            
            # 重置该游戏类型的已用示例
            self.used_examples[game_type] = set()
            self.pattern_results[game_type] = {}
            
            # 运行0-shot到2-shot实验
            for shot, shot_patterns in patterns.items():
                self.pattern_results[game_type][shot] = {}
                
                for pattern in shot_patterns:
                    results = self.run_pattern_experiment(game_type, pattern, num_trials)
                    
                    # 计算准确率
                    correct = sum(1 for r in results if r["correct"])
                    total = len(results)
                    accuracy = correct / total if total > 0 else 0
                    
                    self.pattern_results[game_type][shot][pattern] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "results": results
                    }
                    
                    # 保存中间结果
                    self.save_pattern_results()
            
            # 确定最佳2-shot模式
            best_2shot = max(
                self.pattern_results[game_type]["2-shot"].items(),
                key=lambda x: x[1]["accuracy"]
            )[0]
            
            # 获取最佳2-shot的示例
            best_examples = self.find_examples_by_pattern(game_type, best_2shot)
            
            # 运行3-shot实验
            self.pattern_results[game_type]["3-shot"] = {}
            
            # 添加成功示例
            success_examples = self.find_examples_by_pattern(game_type, "best_plus_success", 1)
            if success_examples:
                results = self.run_pattern_experiment(game_type, "best_plus_success", num_trials, 
                                                    additional_examples=best_examples)
                self.pattern_results[game_type]["3-shot"]["best_plus_success"] = {
                    "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0,
                    "correct": sum(1 for r in results if r["correct"]),
                    "total": len(results),
                    "results": results
                }
            
            # 添加失败示例
            failure_examples = self.find_examples_by_pattern(game_type, "best_plus_failure", 1)
            if failure_examples:
                results = self.run_pattern_experiment(game_type, "best_plus_failure", num_trials,
                                                    additional_examples=best_examples)
                self.pattern_results[game_type]["3-shot"]["best_plus_failure"] = {
                    "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0,
                    "correct": sum(1 for r in results if r["correct"]),
                    "total": len(results),
                    "results": results
                }
            
            # 保存最终结果
            self.save_pattern_results()
    
    def save_pattern_results(self):
        """Save pattern-based experiment results"""
        results_file = self.result_dir / "pattern_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.pattern_results, f, ensure_ascii=False, indent=2)
        
        # Generate summary report
        summary_file = self.result_dir / "pattern_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"===== Pattern-based Experiment Results =====\n")
            f.write(f"Model: {self.model_name}\n\n")
            
            for game_type, shot_results in self.pattern_results.items():
                f.write(f"\n=== {game_type} ===\n")
                for shot, pattern_results in shot_results.items():
                    f.write(f"\n{shot}:\n")
                    for pattern, results in pattern_results.items():
                        f.write(f"  {pattern}: {results['accuracy']:.4f} ({results['correct']}/{results['total']})\n")

def main():
    parser = argparse.ArgumentParser(description="Run pattern-based physical intuition experiments")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--model_name", type=str, default="gemma3:27b", 
                        help="Model name (options: gemma3:27b, llama3.2-vision:11b, minicpm-v:8b)")
    parser.add_argument("--base_url", type=str, default="http://localhost:11434/v1", help="Ollama API base URL")
    parser.add_argument("--api_key", type=str, default="ollama", help="API key")
    parser.add_argument("--output_dir", type=str, default="results_pattern", help="Output directory")
    parser.add_argument("--prompt_dir", type=str, default="prompt1", help="Prompt directory")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials per pattern")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    
    # Create evaluator
    evaluator = PatternBasedEvaluator(
        data_root=args.data_root,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir
    )
    
    # Run pattern-based experiments
    evaluator.run_all_patterns(num_trials=args.num_trials)

if __name__ == "__main__":
    main() 