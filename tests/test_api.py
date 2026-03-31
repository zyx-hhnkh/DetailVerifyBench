"""
API 连通性测试脚本
用法: python test_api.py [model_name]
示例: python test_api.py gpt-5.2
      python test_api.py gpt-5.1
      python test_api.py gpt-5.4
      python test_api.py gemini-3-pro
      python test_api.py gemini-3.1-pro
      python test_api.py opus-4.6
      python test_api.py all
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv(override=True)

def test_openai(model_name):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        print(f"  [FAIL] OPENAI_API_KEY not set")
        return False

    client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 300}
    if base_url:
        client_kwargs["base_url"] = base_url
    print(f"  Base URL: {base_url or 'default (api.openai.com)'}")

    client = OpenAI(**client_kwargs)
    try:
        # 纯文本测试（不带图片）
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_completion_tokens=16,
        )
        content = response.choices[0].message.content.strip()
        print(f"  [OK] Response: {content}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_gemini(model_name):
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    if not api_key:
        print(f"  [FAIL] GEMINI_API_KEY not set")
        return False

    gemini_model_id = {
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
    }[model_name]

    print(f"  Base URL: {base_url or 'default'}")
    print(f"  Model ID: {gemini_model_id}")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["http_options"] = {"base_url": base_url}

    client = genai.Client(**client_kwargs)
    try:
        response = client.models.generate_content(
            model=gemini_model_id,
            contents=["Say 'hello' and nothing else."],
            config=types.GenerateContentConfig(max_output_tokens=16),
        )
        content = response.text.strip() if response.text else "(empty)"
        print(f"  [OK] Response: {content}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_anthropic(model_name):
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not api_key:
        print(f"  [FAIL] ANTHROPIC_API_KEY not set")
        return False

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    print(f"  Base URL: {base_url or 'default (api.anthropic.com)'}")

    client = anthropic.Anthropic(**client_kwargs)
    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16,
            temperature=0.01,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        content = response.content[0].text.strip()
        print(f"  [OK] Response: {content}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    # 设置代理
    if os.getenv("HTTP_PROXY"):
        os.environ["http_proxy"] = os.getenv("HTTP_PROXY")
    if os.getenv("HTTPS_PROXY"):
        os.environ["https_proxy"] = os.getenv("HTTPS_PROXY")
    print(f"Proxy: {os.getenv('http_proxy', 'not set')}\n")

    models = {
        # "gpt-5.1": test_openai,
        "gpt-5.2": test_openai,
        "gpt-5.4": test_openai,
        "gemini-3-pro": test_gemini,
        "gemini-3-flash": test_gemini,
        "gemini-3.1-pro": test_gemini,
        "opus-4.6": test_anthropic,
    }

    if target == "all":
        test_list = list(models.items())
    elif target in models:
        test_list = [(target, models[target])]
    else:
        print(f"Unknown model: {target}")
        print(f"Available: {', '.join(models.keys())}, all")
        sys.exit(1)

    for name, test_fn in test_list:
        print(f"Testing {name} ...")
        test_fn(name)
        print()

if __name__ == "__main__":
    main()
