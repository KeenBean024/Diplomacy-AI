# Diplomacy-AI

## Getting Started
1. Create .env file
```text
OPENAI_API_KEY="..."
GEMINI_API_KEY="..."
```
2. Install packages
```bash
# Activate environment
pip install -f requirements.txt
```

3. Start VLLM Llama-3.2-1B model
```bash
docker run --runtime nvidia --gpus all -v <LOCAL_DIRECTORY_PATH>:/root/.cache/huggingface -p 8000:8000 --ipc=host --env "HUGGING_FACE_HUB_TOKEN=<HF_TOKEN>" vllm/vllm-openai:latest --model meta-llama/Llama-3.2-1B-Instruct
```