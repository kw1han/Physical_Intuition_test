#!/usr/bin/env python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

print("测试Ollama API连接...")

try:
    response = client.chat.completions.create(
        model="gemma3:27b",
        messages=[
            {"role": "user", "content": "What is your name?"}
        ],
        max_tokens=50
    )
    print(f"响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"错误: {e}") 