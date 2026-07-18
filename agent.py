import os
import sys
import json
import requests
import re
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://knot.woa.com/apigw/api/v1/agents/agui/e686a019f9ef40f899df3e2127e93310"
KNOT_API_TOKEN = os.getenv("KNOT_API_TOKEN")
KNOT_API_USER = os.getenv("KNOT_API_USER")

headers = {
    "x-knot-api-token": KNOT_API_TOKEN,
    "x-knot-api-user": KNOT_API_USER,
}


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: 只允许数字和 + - * / ( )"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def call_knot_api(message: str, conversation_id: str = "") -> tuple:
    chat_body = {
        "input": {
            "message": message,
            "conversation_id": conversation_id,
            "model": "deepseek-v3.1",
            "stream": True,
            "enable_web_search": False,
            "chat_extra": {
                "agent_client_uuid": "",
                "attached_images": [],
                "extra_headers": {},
                "background_knowledge": "",
                "enable_thinking": False,
                "max_context_tokens": 200000,
                "reasoning_effort": "medium",
            },
            "temperature": 0.5,
        }
    }

    response = requests.post(API_URL, json=chat_body, headers=headers, stream=True)
    response.raise_for_status()

    full_content = ""
    current_conversation_id = conversation_id

    for chunk in response.iter_lines():
        if not chunk:
            continue
        chunk_str = chunk.decode("utf-8").lstrip("data:").strip()
        if chunk_str == "[DONE]":
            break
        try:
            msg = json.loads(chunk_str)
        except json.JSONDecodeError:
            continue

        if "type" not in msg:
            continue

        msg_type = msg["type"]

        if "rawEvent" in msg:
            event = msg["rawEvent"]
            if "conversation_id" in event and event["conversation_id"]:
                current_conversation_id = event["conversation_id"]

        if msg_type == "TEXT_MESSAGE_CONTENT":
            content = msg["rawEvent"].get("content", "")
            full_content += content
            print(content, end="")

        elif msg_type == "THINKING_TEXT_MESSAGE_CONTENT":
            content = msg["rawEvent"].get("content", "")
            print(f"\n[思考] {content}", end="")

        elif msg_type == "TOOL_CALL_START":
            event = msg["rawEvent"]
            tool_name = event.get("tool_name", "")
            tool_input = event.get("tool_input", {})
            print(f"\n[调用工具] {tool_name}({tool_input})")

        elif msg_type == "TOOL_CALL_RESULT":
            event = msg["rawEvent"]
            result = event.get("result", "")
            print(f"\n[工具结果] {result}")

    print()
    return full_content, current_conversation_id


def run_agent():
    conversation_id = ""
    print("=" * 50)
    print("欢迎使用 Knot AGUI Agent")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    while True:
        print("\n" + "-" * 30)
        user_input = input("你: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break

        if not user_input:
            print("请输入内容")
            continue

        print("\nAgent:", end="")
        _, conversation_id = call_knot_api(user_input, conversation_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print("\n=== 最终回答 ===")
        call_knot_api(q)
    else:
        run_agent()