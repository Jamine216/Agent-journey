"""
最小 ReAct Agent · DeepSeek 版(from scratch,无框架)
运行: uv run agent.py
原理: LLM(决策) + 工具(执行) + 循环(多步推理)
模型自己决定「调不调工具、调哪个、结果够不够」,不够就再转一圈。
"""
import os
import re
import json
import sys
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# DeepSeek 官方 OpenAI 兼容地址(注意:不要加 /v1)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# -------工具集 -------
def calculator(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"
    
def get_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    try:
        with open(path.strip(), "r", encoding="utf-8") as f:
            return f.read(2000)
    except Exception as e:
        return f"Error: 无法读取文件 {path}: {e}"


TOOL_FUNCS = {
    "calculator": calculator,
    "get_time": get_time,
    "read_file": read_file,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算算术表达式，支持 + - * / 和括号",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "返回当前本地时间，无需参数",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件内容，最多2000字符",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


SYSTEM_PROMPT = """你是一个能使用工具的助手。

可用工具：
- calculator: 计算算术表达式，例如 Action Input: 23 * 7 + 5
- get_time: 返回当前时间，无需参数，例如 Action Input: （留空）
- read_file: 读取一个文本文件，输入文件路径，例如 Action Input: notes.txt

当需要工具时，严格按格式输出：
Thought: 你的思考
Action: <工具名>
Action Input: <参数，没有就留空>

当你已经有最终答案时，严格按格式结束：
Final Answer: <给用户的答案>

注意：每步只调用一个工具。调用后等待 Observation 再继续。
"""


def parse_action(text: str):
    # 1) 优先识别 Final Answer
    m_fa = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if m_fa:
        return "FINAL", m_fa.group(1).strip()
    # 2) 否则解析 Action / Action Input
    m = re.search(r"Action:\s*(\w+)\s*Action Input:\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def run_agent(user_input: str, max_turns: int = 5, repeat_limit: int = 2):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    seen = {} # (action, input) -> 次数，用于循环保护
    for turn in range(1, max_turns + 1):
        print(f"\n--- 第 {turn} 轮推理 ---")
        resp = client.chat.completions.create(
            # ⚠️ 用 deepseek-chat:V3 支持稳定 tool calling
            #    deepseek-reasoner(R1)不支持稳定工具调用,别在这里用
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
       
        content = msg.content or ""
        print(f"模型输出: {content}")

        action, action_input = parse_action(content)

        # 保护①：明确的最终答案 → 干净结束
        if action == "FINAL":
            return action_input
            
        if action is None:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "你的输出不符合格式。请严格用 'Action: <工具名>' + 'Action Input: <参数>'，或 'Final Answer: <答案>'。",
            })
            continue

        key = (action, action_input)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > repeat_limit:
            return (f"⚠️ 检测到工具 '{action}' 被重复调用（参数：{action_input}），"
                    f"已停止以避免死循环。")

        if action not in TOOL_FUNCS:
            return f"未知工具: {action}（可用：{', '.join(TOOL_FUNCS.keys())}）"

        print(f"[调用工具] {action}({action_input})")
        result = TOOL_FUNCS[action](action_input)
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"Observation: {result}",
        })

    return "达到最大步数仍未得出最终答案。"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "现在几点了？另外帮我算一下 88 * 9 等于多少？"
    print("\n=== 最终回答 ===")
    print(run_agent(q))