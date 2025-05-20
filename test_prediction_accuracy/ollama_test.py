#!/usr/bin/env python
import base64
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="ollama",  # required but unused for Ollama
    base_url="http://localhost:11434/v1",
)

# 测试纯文本查询
def test_text_model(model_name):
    print(f"\n测试纯文本模型: {model_name}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you tell me what model you are?"}
            ],
            max_tokens=100
        )
        print(f"响应: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

# 测试多模态查询
def test_vision_model(model_name):
    print(f"\n测试多模态模型: {model_name}")
    try:
        # 加载一个示例图像（使用balanced_dataset中的一个场景初始帧）
        import glob
        image_paths = glob.glob("balanced_dataset/Subj_1/*/frame_0000.png")
        if not image_paths:
            print("找不到测试图像")
            return False
            
        image_path = image_paths[0]
        print(f"使用图像: {image_path}")
        
        # 编码图像为base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 创建多模态查询
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can analyze images."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        print(f"响应: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    # 测试纯文本模型
    text_model = "gemma3:27b"
    test_text_model(text_model)
    
    # 测试多模态模型
    vision_models = ["llama3.2-vision", "minicpm-v"]
    for model in vision_models:
        test_vision_model(model) 