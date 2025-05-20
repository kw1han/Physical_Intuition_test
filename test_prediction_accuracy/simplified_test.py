#!/usr/bin/env python
import os
import base64
from pathlib import Path
from openai import OpenAI
import random
import json
import time

# 初始化Ollama客户端
client = OpenAI(
    api_key="ollama",  # required but unused for Ollama
    base_url="http://localhost:11434/v1",
)

# 测试场景路径
DATA_ROOT = "balanced_dataset"

def find_random_scene(data_root=DATA_ROOT, count=1):
    """查找随机的测试场景"""
    subjects = [d for d in Path(data_root).iterdir() if d.is_dir() and d.name.startswith("Subj_")]
    
    if not subjects:
        print(f"错误：在 {data_root} 中找不到主题文件夹")
        return []
    
    all_scenes = []
    for subject in subjects:
        scenes = [d for d in subject.iterdir() if d.is_dir()]
        all_scenes.extend(scenes)
    
    if not all_scenes:
        print(f"错误：在主题文件夹中找不到场景")
        return []
    
    return random.sample(all_scenes, min(count, len(all_scenes)))

def test_model(model_name, scene_path):
    """测试模型在单个场景上的表现"""
    print(f"\n测试模型 {model_name} 在场景 {scene_path.name} 上的表现")
    
    # 获取场景的初始帧
    try:
        first_frame = next(scene_path.glob("frame_0000.png"))
    except StopIteration:
        print(f"错误：在 {scene_path} 中找不到初始帧")
        return None
    
    # 确定真实结果
    true_result = scene_path.name.endswith("True")
    
    # 编码图像为base64
    with open(first_frame, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # 构建提示
    messages = [
        {
            "role": "system", 
            "content": "You are an AI assistant with strong physical intuition. You need to predict whether the red ball will reach the green target area based on the physical scene image. Start your answer with a clear YES or NO before providing explanation."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the initial scene. Will the red ball successfully reach the green target?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }
    ]
    
    try:
        # 调用模型
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=300,
            temperature=0.0
        )
        
        # 解析响应
        response_text = response.choices[0].message.content
        prediction = parse_prediction(response_text)
        
        # 记录结果
        result = {
            "scene": scene_path.name,
            "model": model_name,
            "true_result": true_result,
            "prediction": prediction,
            "correct": prediction == true_result if prediction is not None else None,
            "response": response_text
        }
        
        print(f"真实结果: {'成功' if true_result else '失败'}")
        print(f"预测结果: {'成功' if prediction else '失败'} (正确: {result['correct']})")
        print(f"模型回答: {response_text[:100]}...")
        
        return result
    
    except Exception as e:
        print(f"错误: {e}")
        return None

def parse_prediction(response_text):
    """从模型响应中解析预测结果"""
    text = response_text.lower().strip()
    
    # 简单方法：检查YES/NO关键词
    if text.startswith("yes"):
        return True
    elif text.startswith("no"):
        return False
    
    # 备选方法：检查成功/失败相关词汇
    success_count = sum(1 for word in ["success", "reach", "will reach", "successfully"] if word in text)
    failure_count = sum(1 for word in ["fail", "won't reach", "will not reach", "cannot reach"] if word in text)
    
    if success_count > failure_count:
        return True
    elif failure_count > success_count:
        return False
    
    # 如果无法确定，返回None
    print(f"警告: 无法从响应中确定预测结果:\n{response_text}")
    return None

if __name__ == "__main__":
    # 获取随机的测试场景
    scenes = find_random_scene(count=3)
    
    if not scenes:
        print("错误：找不到测试场景")
        exit(1)
    
    # 要测试的模型
    models = ["gemma3:27b", "llama3.2-vision", "minicpm-v"]
    
    # 存储结果
    results = []
    
    # 测试每个模型在每个场景上的表现
    for model in models:
        for scene in scenes:
            result = test_model(model, scene)
            if result:
                results.append(result)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"ollama_test_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试完成，结果已保存到 ollama_test_results_{timestamp}.json")
    
    # 简单统计
    if results:
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # 按模型统计
        model_stats = {}
        for r in results:
            model = r["model"]
            if model not in model_stats:
                model_stats[model] = {"correct": 0, "total": 0}
            
            model_stats[model]["total"] += 1
            if r["correct"]:
                model_stats[model]["correct"] += 1
        
        print(f"\n总体准确率: {accuracy:.2f} ({correct_count}/{total_count})")
        print("\n按模型统计:")
        for model, stats in model_stats.items():
            model_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {model}: {model_accuracy:.2f} ({stats['correct']}/{stats['total']})") 