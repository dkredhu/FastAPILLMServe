import requests

API_URL = "http://localhost:8080"

def call_generate(prompt, context="", thinking_mode=False):
    payload = {
        "prompt": prompt,
        "context": context,
        "max_new_tokens": 64,
        "temperature": 1.0,
        "thinking_mode": thinking_mode
    }
    resp = requests.post(f"{API_URL}/generate", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def call_generate_vllm(prompt, context="", thinking_mode=False):
    payload = {
        "prompt": prompt,
        "context": context,
        "max_new_tokens": 64,
        "temperature": 1.0,
        "thinking_mode": thinking_mode
    }
    resp = requests.post(f"{API_URL}/generate_vllm", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

if __name__ == "__main__":
    print("Calling /generate endpoint:")
    call_generate("What is a large language model?", thinking_mode=True)

    print("\nCalling /generate_vllm endpoint:")
    call_generate_vllm("What is a large language model?", thinking_mode=True)

    # Example with context
    print("\nCalling /generate endpoint with context:")
    call_generate(
        prompt="How does it relate to artificial intelligence?",
        context="Large language models are a type of AI model trained on vast amounts of text data.",
        thinking_mode=True
    )
