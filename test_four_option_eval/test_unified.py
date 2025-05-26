import os
import json
import random
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import time
import base64
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import ollama
except ImportError:
    ollama = None

class Logger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_subdir = output_dir / f"evaluation_log_{timestamp}"
        self.log_subdir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_subdir / f"evaluation_log_{timestamp}.txt"
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    def log(self, message: str):
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    def log_section(self, title: str):
        separator = "=" * 50
        message = f"\n{separator}\n{title}\n{separator}\n"
        self.log(message)
    def log_subsection(self, title: str):
        separator = "-" * 30
        message = f"\n{separator}\n{title}\n{separator}\n"
        self.log(message)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class PhysicalIntuitionEvaluator:
    def __init__(self,
                data_root: str,
                model_name: str = "gpt-4o",
                api_key: Optional[str] = None,
                base_url: Optional[str] = None,
                output_dir: str = "results",
                game_type: Optional[str] = None,
                max_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.9,
                #top_k: int = 50,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                n: int = 1,
                stream: bool = False):
        #self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n
        self.stream = stream
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.game_type = game_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_dir = Path("/home/student0/Physical_Intuition_test/prompt")
        self.logger = Logger(self.output_dir)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = self._init_client()
        self.results = []
        self.response_times = []
        self.subjects = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("Subj_")])
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

    def _init_client(self):
        # Ollama优先（本地推理）
        if self.base_url and "localhost" in self.base_url or self.api_key == "ollama":
            #if OpenAI is not None:
            #    return OpenAI(
            #        api_key=self.api_key or "ollama",
            #        base_url=self.base_url,
            #        default_headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            #    )
            #elif ollama is not None:
            #    return ollama.Client(host=self.base_url)
            #else:
            #    raise ImportError("Neither openai nor ollama package is available.")
            if ollama is not None:
                self.logger.log("Using Ollama local client...")
                # 去除/v1后缀，使用ollama原生端口
                ollama_host = self.base_url.replace("/v1", "") if self.base_url else "http://localhost:11434"
                return ollama.Client(host=ollama_host)
            else:
                raise ImportError("Ollama package is not available.")
        # OpenAI/DeepSeek等兼容API
        if OpenAI is not None:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            )
        raise ImportError("openai package is required for this script.")

    def find_trial_folders(self, success_only: bool = None, game_types: List[str] = None) -> List[Path]:
        all_trials = []
        for subject in self.subjects:
            for trial_folder in subject.iterdir():
                if not trial_folder.is_dir():
                    continue
                parts = trial_folder.name.split('_')
                if len(parts) < 5:
                    continue
                game_type = parts[0]
                is_success = parts[-1] == "True"
                if game_types and game_type not in game_types:
                    continue
                if success_only is not None and is_success != success_only:
                    continue
                all_trials.append(trial_folder)
        return all_trials

    def _call_deepseek_api(self, messages):
        """专门处理 DeepSeek API 的调用"""
        self.logger.log("Using DeepSeek API...")
        
        # 提取系统消息和用户消息
        system_msg = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
        user_msg = next((msg for msg in messages if msg['role'] == 'user'), None)
        
        if not user_msg:
            raise ValueError("No user message found")
            
        # 构建 DeepSeek 消息格式
        deepseek_message = {
            "role": "user",
            "content": []
        }
        
        # 添加系统消息作为文本
        if system_msg:
            deepseek_message["content"].append({
                "type": "text",
                "text": system_msg
            })
        
        # 处理用户消息
        if isinstance(user_msg['content'], list):
            # 处理多模态内容
            for item in user_msg['content']:
                if item['type'] == 'text':
                    deepseek_message["content"].append({
                        "type": "text",
                        "text": item['text']
                    })
                elif item['type'] == 'image_url':
                    image_url = item['image_url']['url']
                    if image_url.startswith('data:image/png;base64,'):
                        deepseek_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        })
                    else:
                        deepseek_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_url}"
                            }
                        })
        else:
            # 处理纯文本消息
            deepseek_message["content"].append({
                "type": "text",
                "text": user_msg['content']
            })
        
        # 构建请求数据
        request_data = {
            "model": self.model_name,
            "messages": [deepseek_message],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }
        
        # 调试日志
        self.logger.log("\nDeepSeek Request Data:")
        self.logger.log(f"Model: {request_data['model']}")
        self.logger.log(f"Message content types: {[item['type'] for item in request_data['messages'][0]['content']]}")
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_data
        )
        
        if response.status_code != 200:
            error_msg = response.text
            self.logger.log(f"DeepSeek API Error: {error_msg}")
            raise Exception(f"Error code: {response.status_code} - {error_msg}")
        
        response_json = response.json()
        return {
            "choices": [{
                "message": {
                    "content": response_json["choices"][0]["message"]["content"]
                }
            }]
        }

    def call_model(self, messages: List[Dict]) -> Dict:
        max_retries = 3
        retry_delay = 3
        for attempt in range(max_retries):
            try:
                self.logger.log(f"\nAttempt {attempt+1}/{max_retries} to call API with model: {self.model_name}")
                self.logger.log(f"Using API URL: {getattr(self.client, 'base_url', str(self.client))}")
                if attempt == 0:
                    self.logger.log(f"API key (first 5 chars): {str(self.api_key)[:5]}...")
                    self.logger.log(f"Number of messages: {len(messages)}")
                    self.logger.log(f"First message role: {messages[0]['role']}")

                # DeepSeek API
                if "deepseek" in self.model_name.lower():
                    return self._call_deepseek_api(messages)
                
                # Ollama原生客户端
                elif isinstance(self.client, ollama.Client):
                    self.logger.log("Using Ollama native API...")
                    # 转换消息格式为ollama可以理解的格式
                    ollama_messages = []
                    
                    # # 打印原始消息内容
                    # self.logger.log("\n=== Original Messages ===")
                    # for msg in messages:
                    #     self.logger.log(f"\nRole: {msg['role']}")
                    #     if isinstance(msg['content'], list):
                    #         self.logger.log("Content (list):")
                    #         for item in msg['content']:
                    #             if item['type'] == 'text':
                    #                 self.logger.log(f"Text: {item['text']}")
                    #             elif item['type'] == 'image_url':
                    #                 self.logger.log("Image: [base64 data]")
                    #     else:
                    #         self.logger.log(f"Content: {msg['content']}")
                    
                    for msg in messages:
                        if msg['role'] == 'system':
                            # 系统消息直接转换
                            ollama_messages.append({
                                'role': 'system',
                                'content': msg['content']
                            })
                        elif msg['role'] == 'user':
                            content = msg['content']
                            if isinstance(content, list):
                                # 处理包含图片的复杂内容
                                main_prompt = None
                                scene_messages = []
                                
                                # 遍历内容列表，收集文本和图片
                                for i, item in enumerate(content):
                                    if item['type'] == 'text':
                                        if main_prompt is None:
                                            main_prompt = item['text']  # 第一个文本是主提示
                                        else:
                                            # 存储场景描述文本
                                            scene_messages.append({
                                                'text': item['text'],
                                                'image': None
                                            })
                                    elif item['type'] == 'image_url':
                                        # 将图片添加到最后一个场景消息中
                                        if scene_messages:
                                            scene_messages[-1]['image'] = item['image_url']['url'].split(',')[1]
                                
                                # 添加主提示
                                if main_prompt:
                                    ollama_messages.append({
                                        'role': 'user',
                                        'content': main_prompt
                                    })
                                
                                # 为每个场景创建单独的消息
                                for scene in scene_messages:
                                    if scene['text'] and scene['image']:
                                        ollama_messages.append({
                                            'role': 'user',
                                            'content': scene['text'],
                                            'images': [scene['image']]
                                        })
                                
                                # 添加最后的提示
                                # ollama_messages.append({
                                #     'role': 'user',
                                #     'content': "Based on the above scenes, which scene (A, B, C, or D) do you predict will succeed? Please provide your reasoning and final answer in the format specified."
                                # })
                            else:
                                # 简单文本内容
                                ollama_messages.append({
                                    'role': 'user',
                                    'content': content
                                })
                    
                    response = self.client.chat(model=self.model_name, messages=ollama_messages)
                    self.logger.log("Ollama call successful!")
                    return response
                
                # OpenAI兼容API（包括其他兼容OpenAI接口的模型）
                # ✅ OpenAI兼容API（包括 DashScope）
                elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                    stream_required = "qwen" in self.model_name.lower()  # Qwen 系列要求开启 stream
                    request_data = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty,
                        "n": self.n,
                        "stream": self.stream or stream_required  # 覆盖某些模型强制stream=True
                    }
                    response = self.client.chat.completions.create(**request_data)

                    if stream_required:
                        # 🔁 拼接流式返回内容
                        full_content = ""
                        for chunk in response:
                            delta = getattr(chunk.choices[0].delta, "content", "") or ""
                            full_content += delta
                        self.logger.log("Streamed response received.")
                        return {"choices": [{"message": {"content": full_content}}]}
                    else:
                        self.logger.log("API call successful!")
                        return response

                else:
                    raise RuntimeError("Unknown client type.")

            except Exception as e:
                error_msg = str(e)
                self.logger.log(f"API调用错误 (Attempt {attempt+1}/{max_retries}): {error_msg}")
                if "invalid_api_key" in error_msg or "401" in error_msg:
                    self.logger.log("API密钥无效。请检查API密钥是否正确，或尝试使用环境变量OPENAI_API_KEY设置密钥。")
                elif "rate_limit" in error_msg:
                    self.logger.log("达到API速率限制，将增加重试延迟时间")
                    retry_delay *= 2
                if attempt < max_retries - 1:
                    self.logger.log(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)

        self.logger.log("所有重试尝试都失败，返回None")
        return None

    def save_detailed_analysis(self, game_type_results, test_groups=None):
        """保存详细分析结果"""
        analysis_file = self.logger.log_subdir / 'physical_intuition_analysis.txt'
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

    def run_evaluation(self, num_sets: int = 5, repetitions: int = 4):
        self.results = []
        self.response_times = []
        valid_scenes = self.find_same_scene_trials()
        if not valid_scenes:
            self.logger.log("No valid scenes found (require at least 3 failure cases and 1 success case in the same scene)")
            return
        #scenes_to_test = valid_scenes[:num_sets]
        #self.logger.log(f"\nTesting {len(scenes_to_test)} scenes (limited by num_sets={num_sets})")
        
        total_sets = 0
        for failure_trials, success_trials in valid_scenes:
            #if total_sets >= num_sets:  # 移到外层循环
            #    break  # 使用 break 而不是 return
            
            self.logger.log(f"\nStarting evaluation for scene: {success_trials[0].name.split('_')[0]}_{success_trials[0].name.split('_')[3]}")
            self.logger.log(f"This scene has {len(success_trials)} success cases")
            current_game_type = success_trials[0].name.split('_')[0]
            
            for success_case in success_trials:
                #if total_sets >= num_sets:  # 内层循环也检查
                #    break
                
                selected_failures = random.sample(failure_trials, 3)
                all_cases = selected_failures + [success_case]
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
                if len(valid_cases) < 4:
                    self.logger.log(f"Warning: Not enough valid cases in the current combination (need 4), skipping")
                    continue
                all_cases = valid_cases
                original_correct_index = all_cases.index(success_case) + 1
                consistency_results = []
                self.logger.log(f"\n----- Starting {repetitions} repetition tests for the same image set -----")
                target_positions = list(range(4))
                if repetitions > 4:
                    extra_positions = [random.randint(0, 3) for _ in range(repetitions - 4)]
                    target_positions.extend(extra_positions)
                random.shuffle(target_positions)
                for rep in range(repetitions):
                    target_position = target_positions[rep]
                    cases_without_success = [case for case in all_cases if case != success_case]
                    random.shuffle(cases_without_success)
                    shuffled_cases = cases_without_success.copy()
                    shuffled_cases.insert(target_position, success_case)
                    shuffled_images = []
                    for case in shuffled_cases:
                        idx = all_cases.index(case)
                        shuffled_images.append(valid_images[idx])
                    correct_index = shuffled_cases.index(success_case) + 1
                    random_indices = []
                    for case in shuffled_cases:
                        random_indices.append(all_cases.index(case))
                    letters = ['A', 'B', 'C', 'D']
                    prompt_text = self.load_prompt_from_file(current_game_type)
                    messages = []
                    system_message = {
                        "role": "system",
                        "content": "You are an AI assistant with strong physical intuition. You need to analyze four physical scene images in their initial states and determine which scene will allow the red ball to successfully reach the green target area. These scenes come from the same physical environment setup but with slightly different initial conditions. Carefully observe the details of object positions, orientations, and obstacle distributions in each scene. Important note: Scenes are labeled with letters A-D, and the order has no correlation with success probability. Please make judgments completely based on physical principles. Structure your answer with a detailed reasoning for each scene followed by a final result statement."
                    }
                    messages.append(system_message)
                    prompt_content = [
                        {"type": "text", "text": prompt_text}
                    ]
                    for i, base64_img in enumerate(shuffled_images):
                        prompt_content.extend([
                            {"type": "text", "text": f"Scene {letters[i]}:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                        ])
                    messages.append({"role": "user", "content": prompt_content})
                    messages.append({
                        "role": "system",
                        "content": "Remember to end your response with exactly: Final Result: \"I predict that scene [A/B/C/D] will succeed.\" "
                    })
                    start_time = time.time()
                    response = self.call_model(messages)
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    if not response:
                        continue
                    if hasattr(response, 'choices'):
                        response_text = response.choices[0].message.content
                    elif hasattr(response, 'message'):
                        response_text = response.message['content']
                    elif isinstance(response, dict) and 'choices' in response:
                        response_text = response['choices'][0]['message']['content']
                    else:
                        response_text = str(response)
                    predicted_index = self.extract_predicted_scene_number(response_text)
                    original_predicted_index = -1
                    if predicted_index > 0:
                        predicted_case_index = random_indices[predicted_index - 1]
                        original_predicted_index = predicted_case_index + 1
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
                    self.logger.log(f"\n----- Test Set {len(self.results)} (Repetition {rep+1}/{repetitions}) -----")
                    self.logger.log(f"Success case: {success_case.name}")
                    self.logger.log(f"Scene type: {self.game_type_desc.get(result['game_type'], result['game_type'])}")
                    self.logger.log(f"Scene ID: {result['scene_id']}")
                    self.logger.log(f"Correct scene position (A-D): {letters[correct_index-1]}")
                    self.logger.log(f"Shuffle mapping: {result['shuffle_mapping']}")
                    self.logger.log(f"Model prediction: {letters[predicted_index-1] if predicted_index > 0 else 'Invalid prediction'}")
                    self.logger.log(f"Prediction {'correct' if result['correct'] else 'incorrect'}")
                    self.logger.log(f"Response time: {response_time:.2f} seconds")
                    self.logger.log("\nImage paths:")
                    for i, case in enumerate(shuffled_cases):
                        frame_path = next(case.glob("frame_0000.png"))
                        self.logger.log(f"Scene {letters[i]}: {frame_path}")
                    self.logger.log("")
                    #self.save_results("physical_intuition_results.json")
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
                # 在每个测试集完成后更新计数
                total_sets += 1
            
        self.logger.log(f"\nReached requested number of test sets ({num_sets}), completing evaluation")
        self.analyze_results()  # 确保在循环结束后调用 analyze_results

    def extract_predicted_scene_number(self, response_text: str) -> int:
        self.logger.log("\n=== Model Response Text ===")
        self.logger.log(response_text)
        self.logger.log("==========================")
        final_result_match = re.search(r"Final Result:.*?\"I predict that scene ([A-D]) will succeed\.\"", response_text, re.DOTALL)
        if final_result_match:
            letter = final_result_match.group(1)
            self.logger.log(f"\nFound prediction in Final Result: {letter}")
            return {'A': 1, 'B': 2, 'C': 3, 'D': 4}[letter]
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
        letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        all_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, response_text)
            for match in matches:
                letter = match.group(1)
                if letter in letter_to_number:
                    all_matches.append((letter, pattern, match.group(0)))
        if all_matches:
            self.logger.log("\nMatches found:")
            for letter, pattern, matched_text in all_matches:
                self.logger.log(f"Letter: {letter}, Pattern: {pattern}, Matched text: {matched_text}")
            return letter_to_number[all_matches[0][0]]
        letters = re.findall(r'\b[A-D]\b', response_text)
        if letters:
            self.logger.log(f"\nFound standalone letter: {letters[0]}")
            return letter_to_number[letters[0]]
        self.logger.log("\nWarning: Could not extract valid scene number from response")
        return -1

    def analyze_results(self):
        if not self.results:
            self.logger.log("No results to analyze")
            return
        total_sets = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['correct'])
        accuracy = correct_predictions / total_sets
        self.logger.log("\n===== Physical Intuition Evaluation Results =====")
        self.logger.log(f"Total test sets: {total_sets}")
        self.logger.log(f"Correct predictions: {correct_predictions}")
        self.logger.log(f"Overall accuracy: {accuracy:.2%}")
        position_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        correct_position_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        for result in self.results:
            if result['predicted_index'] > 0:
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
        self.logger.log("\nConsistency Analysis:")
        test_groups = {}
        for result in self.results:
            group_key = (result['game_type'], result['scene_id'], result['success_case'])
            if group_key not in test_groups:
                test_groups[group_key] = []
            test_groups[group_key].append(result)
        consistency_scores = []
        for group_key, group_results in test_groups.items():
            original_predictions = [r['original_predicted_index'] for r in group_results if r['original_predicted_index'] > 0]
            if not original_predictions:
                continue
            most_common = max(set(original_predictions), key=original_predictions.count)
            consistency_ratio = original_predictions.count(most_common) / len(original_predictions)
            consistency_scores.append(consistency_ratio)
            self.logger.log(f"Group {group_key[0]}_{group_key[1]}: Consistency ratio: {consistency_ratio:.2%}")
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            self.logger.log(f"Average consistency across all groups: {avg_consistency:.2%}")
            perfect_consistent_groups = sum(1 for score in consistency_scores if score == 1.0)
            self.logger.log(f"Groups with perfect consistency: {perfect_consistent_groups}/{len(consistency_scores)} ({perfect_consistent_groups/len(consistency_scores):.1%})")
        avg_response_time = sum(self.response_times) / len(self.response_times)
        min_response_time = min(self.response_times)
        max_response_time = max(self.response_times)
        self.logger.log(f"\nResponse Time Analysis:")
        self.logger.log(f"Average response time: {avg_response_time:.2f} seconds")
        self.logger.log(f"Minimum response time: {min_response_time:.2f} seconds")
        self.logger.log(f"Maximum response time: {max_response_time:.2f} seconds")
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
        error_analysis = self.analyze_error_patterns()
        self.logger.log("\nError Pattern Analysis:")
        for error_type, count in error_analysis.items():
            self.logger.log(f"{error_type}: {count} times")
        filename = f"physical_intuition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_results(filename)
        self.save_detailed_analysis(game_type_results, test_groups)
        self.save_results_to_csv(game_type_results, test_groups)

    def analyze_error_patterns(self) -> Dict[str, int]:
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
        scene_trials = {}
        for subject in self.subjects:
            for trial_folder in subject.iterdir():
                if not trial_folder.is_dir():
                    continue
                parts = trial_folder.name.split('_')
                if len(parts) < 5:
                    continue
                game_type = parts[0]
                if self.game_type and game_type != self.game_type:
                    continue
                scene_id = parts[3]
                is_success = parts[-1] == "True"
                key = (game_type, scene_id)
                if key not in scene_trials:
                    scene_trials[key] = {"success": [], "failure": []}
                if is_success:
                    scene_trials[key]["success"].append(trial_folder)
                else:
                    scene_trials[key]["failure"].append(trial_folder)
        valid_scenes = []
        for (game_type, scene_id), trials in scene_trials.items():
            if len(trials["failure"]) >= 3 and len(trials["success"]) >= 1:
                valid_scenes.append((trials["failure"], trials["success"]))
                self.logger.log(f"Found valid scene: {game_type}_{scene_id}, "
                              f"Failure cases: {len(trials['failure'])}, "
                              f"Success cases: {len(trials['success'])}")
        return valid_scenes

    def save_results(self, filename: str):
        output_path = self.logger.log_subdir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            serializable_results = []
            for result in self.results:
                result_copy = result.copy()
                if 'trial_paths' in result_copy:
                    del result_copy['trial_paths']
                serializable_results.append(result_copy)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        self.logger.log(f"Results saved to: {output_path}\n")

    def save_results_to_csv(self, game_type_results, test_groups=None):
        """将测试结果保存为CSV格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细结果
        detailed_results_file = self.logger.log_subdir / f'detailed_results_{timestamp}.csv'
        detailed_results = []
        letters = ['A', 'B', 'C', 'D']
        
        for result in self.results:
            row = {
                'Test_Set': len(detailed_results) + 1,
                'Repetition': result.get('repetition', 1),
                'Scene_Type': self.game_type_desc.get(result['game_type'], result['game_type']),
                'Scene_ID': result.get('scene_id', ''),
                'Correct_Scene': letters[result['correct_index']-1],
                'Predicted_Scene': letters[result['predicted_index']-1] if result['predicted_index'] > 0 else 'Invalid',
                'Original_Correct_Index': result.get('original_correct_index', 'N/A'),
                'Original_Predicted_Index': result.get('original_predicted_index', 'N/A'),
                'Is_Correct': 'Yes' if result['correct'] else 'No',
                'Response_Time': f"{result['response_time']:.2f}",
                'Success_Case': result.get('success_case', ''),
                'Model_Response': result['response'].replace('\n', ' ').replace(',', ';')
            }
            detailed_results.append(row)
        
        pd.DataFrame(detailed_results).to_csv(detailed_results_file, index=False, encoding='utf-8')
        
        # 2. 保存场景类型分析结果
        scene_analysis_file = self.logger.log_subdir / f'scene_type_analysis_{timestamp}.csv'
        scene_analysis = []
        
        for game_type, stats in game_type_results.items():
            type_accuracy = stats['correct'] / stats['total']
            avg_time = sum(stats['times']) / len(stats['times'])
            row = {
                'Scene_Type': self.game_type_desc.get(game_type, game_type),
                'Total_Cases': stats['total'],
                'Correct_Cases': stats['correct'],
                'Accuracy': f"{type_accuracy:.2%}",
                'Average_Response_Time': f"{avg_time:.2f}"
            }
            scene_analysis.append(row)
        
        pd.DataFrame(scene_analysis).to_csv(scene_analysis_file, index=False, encoding='utf-8')
        
        # 3. 保存一致性分析结果
        if test_groups:
            consistency_file = self.logger.log_subdir / f'consistency_analysis_{timestamp}.csv'
            consistency_analysis = []
            
            for group_key, group_results in test_groups.items():
                original_predictions = [r['original_predicted_index'] for r in group_results if r['original_predicted_index'] > 0]
                if not original_predictions:
                    continue
                
                most_common = max(set(original_predictions), key=original_predictions.count)
                consistency_ratio = original_predictions.count(most_common) / len(original_predictions)
                
                row = {
                    'Scene_Type': group_key[0],
                    'Scene_ID': group_key[1],
                    'Success_Case': group_key[2],
                    'Predictions': str(original_predictions),
                    'Most_Common_Prediction': most_common,
                    'Consistency_Ratio': f"{consistency_ratio:.2%}",
                    'Original_Correct_Index': group_results[0]['original_correct_index']
                }
                consistency_analysis.append(row)
            
            pd.DataFrame(consistency_analysis).to_csv(consistency_file, index=False, encoding='utf-8')
        
        # 4. 保存错误模式分析结果
        error_analysis = self.analyze_error_patterns()
        error_analysis_file = self.logger.log_subdir / f'error_analysis_{timestamp}.csv'
        error_df = pd.DataFrame([{'Error_Type': k, 'Count': v} for k, v in error_analysis.items()])
        error_df.to_csv(error_analysis_file, index=False, encoding='utf-8')
        
        # 5. 保存总体统计结果
        summary_file = self.logger.log_subdir / f'summary_statistics_{timestamp}.csv'
        total_sets = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['correct'])
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        summary_stats = [{
            'Metric': 'Total_Test_Sets',
            'Value': total_sets
        }, {
            'Metric': 'Correct_Predictions',
            'Value': correct_predictions
        }, {
            'Metric': 'Overall_Accuracy',
            'Value': f"{correct_predictions/total_sets:.2%}"
        }, {
            'Metric': 'Average_Response_Time',
            'Value': f"{avg_response_time:.2f}"
        }]
        
        pd.DataFrame(summary_stats).to_csv(summary_file, index=False, encoding='utf-8')
        
        self.logger.log(f"\n测试结果已保存为CSV格式：")
        self.logger.log(f"1. 详细结果：{detailed_results_file}")
        self.logger.log(f"2. 场景类型分析：{scene_analysis_file}")
        self.logger.log(f"3. 一致性分析：{consistency_file}")
        self.logger.log(f"4. 错误模式分析：{error_analysis_file}")
        self.logger.log(f"5. 总体统计：{summary_file}")

    def load_prompt_from_file(self, game_type: str) -> str:
        prompt_file = self.prompt_dir / f"{game_type.lower()}.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            self.logger.log(f"Warning: Prompt file {prompt_file} not found, using default prompt content")
            if game_type == "Bridge":
                return 'Please carefully observe the following four bridge building scenes, where the positions of the blue elongated blocks are different in each scene. Your task is to determine which scene has the most suitable position for the elongated blocks to form a stable bridge structure, helping the red ball successfully reach the green target area.\n\nImportant notes:\n1. Scenes are labeled with letters A-D, and the order has no correlation with success probability\n2. Please focus on:\n   - Whether the elongated blocks can form a stable bridge structure\n   - Whether the bridge structure can support the weight of the ball\n   - Whether the position and angle of the bridge are suitable for the ball to pass\n3. In your analysis, consider:\n   - The support points and balance state of the blocks\n   - The stability of the bridge structure\n   - Whether the ball\'s path will be smooth\n   - Whether there are any structural defects\n\nPlease structure your answer as follows:\nReasoning: For each scene, explain step by step what will happen, whether the red ball will reach the green target area, and why you believe scene X has the highest chance of success.\nFinal Result: "I predict that scene X will succeed."'
            else:
                return f'Please carefully observe the following four scenes, where the positions of objects are slightly different in each scene. Your task is to determine which scene will allow the red ball to successfully reach the green target area.\n\nImportant notes:\n1. Scenes are labeled with letters A-D, and the order has no correlation with success probability\n2. Please analyze each scene based on physical principles\n\nPlease structure your answer as follows:\nReasoning: For each scene, explain step by step what will happen, whether the red ball will reach the green target area, and why you believe scene X has the highest chance of success.\nFinal Result: "I predict that scene X will succeed."'

def main():
    parser = argparse.ArgumentParser(description='Evaluate visual language models\' physical intuition capabilities (Unified Version)')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--api_key', type=str, default=None, help='API key')
    parser.add_argument('--base_url', type=str, default=None, help='API base URL')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--num_sets', type=int, default=5, help='Number of image sets to test')
    parser.add_argument('--game_type', type=str, default=None, help='Game type to test, e.g. Basic')
    parser.add_argument('--repetitions', type=int, default=4, help='Number of repetitions for each image set (for consistency testing)')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter')
    #parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty')
    parser.add_argument('--n', type=int, default=1, help='Number of completions to return')
    parser.add_argument('--stream', type=lambda x: x.lower() == "true", default=False, help='Whether to use streaming')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger = Logger(output_dir)
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
        top_p=args.top_p,
        #top_k=args.top_k,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        n=args.n,
        stream=args.stream
    )
    evaluator.run_evaluation(num_sets=args.num_sets, repetitions=args.repetitions)
    #logger.log(f"Top K: {args.top_k}")
    logger.log(f"Frequency Penalty: {args.frequency_penalty}")
    logger.log(f"Presence Penalty: {args.presence_penalty}")
    logger.log(f"Stream: {args.stream}")
    logger.log(f"Num Completions (n): {args.n}")
    logger.log_section("Evaluation Summary")
    logger.log(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main() 