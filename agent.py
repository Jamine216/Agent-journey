"""
最小 ReAct Agent · DeepSeek 版(from scratch,无框架)
运行: uv run agent.py
原理: LLM(决策) + 工具(执行) + 循环(多步推理)
模型自己决定「调不调工具、调哪个、结果够不够」,不够就再转一圈。
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# DeepSeek 官方 OpenAI 兼容地址(注意:不要加 /v1)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def calculator(expr: str) -> str:
    """计算一个数学表达式,例如 '23*47'。"""
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"


# 工具 schema(OpenAI function-calling 格式)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式,例如 '23*47'。支持 + - * / 和括号。",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    }
]
TOOL_FUNCS = {"calculator": calculator}


def run(user_input: str, max_turns: int = 5):
    messages = [{"role": "user", "content": user_input}]
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
        # 把模型这一轮完整回包(含可能的 tool_calls)存回上下文
        messages.append(msg.model_dump())

        # 模型没调工具 = 它觉得能直接回答了
        if not msg.tool_calls:
            print("最终回答:", msg.content)
            return

        # 模型决定调工具 → 代码执行 → 把结果回填上下文,进入下一轮
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"[调用工具] {tc.function.name}({args})")
            result = TOOL_FUNCS[tc.function.name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    run(
        "一个游戏里 137 个敌人,每个掉 24 金币,总共多少金币?"
        "再乘以 1.15 的暴击加成是多少?"
    )
