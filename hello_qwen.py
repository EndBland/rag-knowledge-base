import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1"
)

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                sys.exit("文件损坏，读取错误")
            """另一个可选方案，备份坏文件后重建
            except json.JSONDecodeError:
                os.rename(HISTORY_FILE, HISTORY_FILE + ".bak")
                return [{"role": "system", ...}]"""
    else:
        return [
            {"role": "system", "content": "你是一个只用一句话回答问题的助手。"}
        ]

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def chat_loop():
    messages = load_history()

    while True:
        user_input = input("你: ")
        if user_input == 'exit':
            break

        messages.append({"role": "user", "content": user_input})

        try:
            completion = client.chat.completions.create(
                model="qwen3.8-max",
                messages=messages
            )

            reply = completion.choices[0].message.content
            print("AI: ", reply)

            messages.append({"role": "assistant", "content": reply})
            save_history(messages)
        except Exception as e:
            print(e)
            messages.pop()

if __name__ == "__main__":
    chat_loop()