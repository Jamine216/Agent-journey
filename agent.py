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


def call_knot_api(message: str) -> str:
    chat_body = {
        "input": {
            "message": message,
            "conversation_id": "",
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
        if msg_type == "TEXT_MESSAGE_CONTENT":
            content = msg["rawEvent"].get("content", "")
            full_content += content
            print(content, end="")

    return full_content


def run_agent(question: str) -> str:
    math_pattern = re.compile(r"[\d+\-*/().\s]+")
    if math_pattern.fullmatch(question.strip()):
        result = calculator(question)
        print(f"本地计算结果: {result}")
        return result

    return call_knot_api(question)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    print("\n=== 最终回答 ===")
    run_agent(q)