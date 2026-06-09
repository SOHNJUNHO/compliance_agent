# vLLM GPU EC2 — Deploy Runbook

The compliance-api Fargate service calls two vLLM endpoints on a single GPU EC2 instance:

| Port | Purpose | Model |
|------|---------|-------|
| 8000 | Chat (LLM) | `Qwen/Qwen3-8B` |
| 8001 | Embeddings | `Qwen/Qwen3-Embedding-0.6B` |

---

## 1. Instance sizing

| Field | Value |
|---|---|
| Instance type | `g5.xlarge` (NVIDIA A10G 24 GB) |
| AMI | Deep Learning AMI GPU PyTorch 2.x (Amazon Linux 2023) |
| Storage | 200 GB gp3 root (models + Docker layers) |
| Subnet | **Private** (no public IP) |
| Security group | Allow `:8000` and `:8001` **from Fargate task SG only** |
| IAM | ECR pull + SSM Session Manager (no SSH needed) |

**Why g5.xlarge?**
- Qwen3-8B in bf16 ≈ 16 GB; A10G 24 GB fits with headroom for KV cache.
- For tighter memory or lower cost, use AWQ 4-bit (≈ 5 GB) — swap the model tag below.

---

## 2. One-time setup on the EC2 instance

```bash
# Install Docker (if not on DL AMI)
sudo yum install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Install nvidia-container-toolkit (DL AMI already has this)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Log in to ECR (optional, for private model images)
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ap-northeast-2.amazonaws.com
```

---

## 3. Launch vLLM chat server (port 8000)

```bash
docker run -d \
  --name vllm-chat \
  --gpus '"device=0"' \
  --runtime nvidia \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 1
```

**Key flags:**
- `--enable-auto-tool-choice --tool-call-parser hermes` — enables OpenAI function-calling
  (LlamaIndex `astructured_predict` routes through tool-calling for `OpenAILike`).
- `--max-model-len 16384` — caps KV cache allocation; adjust up if synthesis prompts are longer.
- For AWQ 4-bit: `--model Qwen/Qwen3-8B-AWQ --quantization awq --dtype half` (≈5 GB VRAM).

---

## 4. Launch vLLM embedding server (port 8001)

```bash
docker run -d \
  --name vllm-embed \
  --gpus '"device=0"' \
  --runtime nvidia \
  -p 8001:8001 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-embedding-0.6B \
  --task embed \
  --port 8001 \
  --tensor-parallel-size 1
```

> **Note:** both containers can share the same GPU (`device=0`). vLLM serialises CUDA
> calls internally. If VRAM is tight, run the embedding server on CPU:
> add `--device cpu` and remove `--gpus`.

---

## 5. Verify endpoints

```bash
# Chat: list models
curl -s http://localhost:8000/v1/models | jq '.data[].id'

# Chat: quick completion
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"hello"}],"max_tokens":16}' \
  | jq '.choices[0].message.content'

# Embed: single embedding
curl -s http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-0.6B","input":"안녕하세요"}' \
  | jq '.data[0].embedding | length'  # expect 1024
```

---

## 6. Environment variables for the Fargate service

Set these in the ECS task definition (plain env) or Secrets Manager (sensitive):

| Variable | Value | Secrets Manager? |
|---|---|---|
| `LLM_API_BASE` | `http://<EC2-private-ip>:8000/v1` | No |
| `LLM_MODEL` | `Qwen3-8B` | No |
| `LLM_API_KEY` | `EMPTY` | No |
| `EMBED_API_BASE` | `http://<EC2-private-ip>:8001/v1` | No |
| `EMBED_MODEL` | `qwen3-embedding-0.6B` | No |
| `EMBED_API_KEY` | `EMPTY` | No |
| `QDRANT_URL` | Qdrant Cloud URL | No |
| `QDRANT_API_KEY` | Qdrant Cloud API key | **Yes** |
| `QDRANT_COLLECTION` | `compliance_agents` | No |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | No |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | **Yes** |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | No |
| `USE_RERANKER` | `1` | No |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | No |

---

## 7. Restart / update

```bash
# Pull latest vLLM image and restart
docker pull vllm/vllm-openai:latest
docker stop vllm-chat vllm-embed
docker rm   vllm-chat vllm-embed
# Re-run steps 3 and 4
```

To update the model without rebuilding: change `--model` in the docker run command.
The HuggingFace cache volume (`~/.cache/huggingface`) persists across container restarts.
