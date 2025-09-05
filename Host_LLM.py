from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()
###Change Qwen 3 model according to your needs
MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Try to import and initialize vLLM if available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        vllm_model = LLM(model=MODEL_NAME, gpu_utilization=0.7, max_model_len=32768)
    else:
        vllm_model = LLM(model=MODEL_NAME, max_model_len=32768)
except ImportError:
    VLLM_AVAILABLE = False
    vllm_model = None

class GenerateRequest(BaseModel):
    context: str = ""
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    thinking_mode: bool = False

class GenerateResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        # Prepare chat template input
        prompt = (
            f"Use the following context to answer the question if provided, otherwise use your own knowledge.\n"
            f"Context: {request.context if request.context else 'N/A'}\n"
            f"Question: {request.prompt}\n"
            f"Answer:"
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        # Use the tokenizer's chat template with thinking mode
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.thinking_mode
        )
        # Tokenize and move to model device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # Generate output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=request.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # Parse thinking content
        try:
            # 151668 is the token id for </think>
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # Return both if thinking_mode, else just content
        if request.thinking_mode:
            result = f"thinking content: {thinking_content}\ncontent: {content}"
        else:
            result = content
        return GenerateResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_vllm", response_model=GenerateResponse)
async def generate_text_vllm(request: GenerateRequest):
    if not VLLM_AVAILABLE or vllm_model is None:
        raise HTTPException(status_code=503, detail="vLLM is not installed or not available on this system.")
    try:
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=1.0,
            do_sample=True,
        )
        messages = [[{"role": "user", "content": f"{request.context}{request.prompt}"}]]
        outputs = vllm_model.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": request.thinking_mode}
        )
        generated = outputs[0].outputs[0].text
        return GenerateResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Host_LLM:app", host="0.0.0.0", port=8080)
