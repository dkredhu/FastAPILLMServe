# FastAPILLMServe

FastAPILLMServe is a robust, production-grade FastAPI server engineered for hosting and serving Qwen3 Large Language Models (LLMs). This solution is designed for scalable, high-performance AI applications and supports state-of-the-art text generation using HuggingFace Transformers and (optionally) vLLM for enhanced throughput.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Model Download](#model-download)
- [Running the Server](#running-the-server)
  - [Local Deployment](#local-deployment)
  - [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)
  - [/generate](#generate)
  - [/generate_vllm](#generate_vllm)
  - [/health](#health) *(if implemented)*
- [Configuration & Parameters](#configuration--parameters)
- [Example Client Usage](#example-client-usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

FastApiLLMServe offers a RESTful API interface for Qwen3 models, enabling users to perform text generation with high efficiency. Two primary endpoints enable different inference modes:
- **HuggingFace Transformers Endpoint:** General-purpose text generation.
- **vLLM Endpoint:** For environments where vLLM is available, providing additional throughput benefits.

Users can easily toggle the "thinking mode" to either include or omit intermediate reasoning content (demarcated by `<think>...</think>`).

---

## Key Features

- **Dual Inference Modes:** Choose between HuggingFace Transformers and optional vLLM.
- **Thinking Mode Toggle:** Control the generation of intermediate reasoning content.
- **Batch Processing:** Handle multiple inference requests concurrently.
- **Dynamic Hardware Utilization:** Auto-detect and utilize GPU resources, with CPU fallback.
- **Containerized Deployment:** Docker-ready for simplified production deployment.
- **Health Monitoring:** (Optional) Endpoint for system health checks.

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- (Optional) CUDA-enabled GPU
- Docker (for containerization)

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/FastApiLLMServe.git
   cd FastApiLLMServe
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:**  
   > To enable vLLM support on a compatible Linux system, install:
   > ```bash
   > pip install vllm
   > ```

---

## Model Download

Download the Qwen3 model and its tokenizer by executing:
```bash
python download_model.py
```
Modify the `MODEL_NAME` variable in both `download_model.py` and `Host_LLM.py` if you wish to use a different Qwen3 model.

---

## Running the Server

### Local Deployment

Start the server locally by running:
```bash
python Host_LLM.py
```
The API will be accessible at [http://0.0.0.0:8080](http://0.0.0.0:8080).

### Docker Deployment

1. **Build the Docker Image:**
   ```bash
   docker build -t fastapi-llm-serve .
   ```

2. **Run the Docker Container:**
   ```bash
   docker run -p 8000:8000 fastapi-llm-serve
   ```
Access the API at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

### /generate (POST)

Generates text using HuggingFace Transformers. It supports configurable context, prompt, token limits, temperature, and thinking mode settings.

**Request Example:**
```json
{
  "context": "Additional details or background information.",
  "prompt": "Explain the significance of large language models.",
  "max_new_tokens": 64,
  "temperature": 1.0,
  "thinking_mode": false
}
```

**Response Example:**
```json
{
  "generated_text": "The model’s response, optionally including thinking content if enabled."
}
```

---

### /generate_vllm (POST)

Generates text using vLLM. This endpoint will return a 503 error if vLLM is not available on the system.

**Request Example:** *(Same as /generate)*

**Response Example:**
```json
{
  "generated_text": "Output generated using vLLM."
}
```

---

### /health (GET) *(Optional)*

Provides the health status of the server, useful for monitoring and orchestration in production environments.

---

## Configuration & Parameters

- **MODEL_NAME:** Configured in both `download_model.py` and `Host_LLM.py` to specify the Qwen3 model.
- **thinking_mode:** A Boolean parameter that toggles the generation of intermediate reasoning output (“thinking mode”). Set to `true` to include reasoning content; otherwise, set to `false`.
- **Batching & Device Utilization:** The server auto-detects GPU availability and falls back to CPU if necessary.

---

## Example Client Usage

Refer to the `example_client.py` file for a sample client that demonstrates how to interact with the API endpoints.  
*Example snippet:*
```python
# ...existing code...
call_generate(
    prompt="How does it relate to artificial intelligence?",
    context="Large language models are a type of AI model trained on vast amounts of text data.",
    thinking_mode=True
)
# ...existing code...
```

---

## Troubleshooting

- **vLLM Endpoint Issues:**  
  Ensure vLLM is installed on a supported platform. If not, the `/generate_vllm` endpoint will respond with a 503 error.

- **Hardware Limitations:**  
  Confirm that your system has compatible drivers (e.g., CUDA for GPU) and sufficient memory for the selected model.

- **Thinking Mode Output:**  
  Validate that your chosen model supports advanced chat templates (with `enable_thinking` functionality) if the output does not meet expectations.

- **Dependency Conflicts:**  
  If you encounter errors, try updating the dependencies:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

---

## References

- [Qwen3 Model Documentation](https://huggingface.co/Qwen)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## License

This project is distributed under the MIT License.
