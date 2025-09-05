from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

def download():
    print(f"Downloading model and tokenizer for {MODEL_NAME} ...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Download complete.")

if __name__ == "__main__":
    download()

