# Setup Guide

Multi-agent compliance Q&A system for Korean securities firms.
This guide covers everything needed to run the project from scratch.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.14+** | Required by `pyproject.toml` |
| **uv** | Package manager — install from [astral.sh/uv](https://astral.sh/uv) |
| **Ollama** | Local LLM server — install from [ollama.com](https://ollama.com) |
| **Qdrant Cloud account** | Free tier available at [cloud.qdrant.tech](https://cloud.qdrant.tech) |
| **Langfuse account** | Free tier available at [cloud.langfuse.com](https://cloud.langfuse.com) |

---

## Step 1 — Install packages

```bash
uv sync
```

This installs all dependencies including LlamaIndex, Langfuse, Qdrant client, and Ollama bindings.

---

## Step 2 — Pull Ollama models

Ollama must be running first:

```bash
ollama serve          # run in a separate terminal and keep it open
```

Then pull the models (one-time, ~5 GB total):

```bash
ollama pull qwen2.5:7b               # LLM (~4.7 GB)
ollama pull qwen3-embedding:0.6b     # embedding model (~0.4 GB)
```

---

## Step 3 — Configure environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set:

```
# Qdrant Cloud — console.qdrant.tech → your cluster → API Keys
QDRANT_URL=https://<cluster-id>.qdrant.tech
QDRANT_API_KEY=<your-qdrant-api-key>

# Langfuse — cloud.langfuse.com → Settings → API Keys
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

`QDRANT_COLLECTION` defaults to `compliance_agents`. `LANGFUSE_HOST` defaults to `https://cloud.langfuse.com`. You only need to set these if you use a different collection name or a self-hosted Langfuse instance.

---

## Step 4 — Ingest data

This parses the raw HTML/XML files in `data/raw/`, embeds all 850 chunks, and uploads them to Qdrant Cloud. Also writes `data/article_lookup.json` for exact-match factchecking.

```bash
python main.py ingest
```

Expected output:
```
=== 1. 수집 ===
INFO scraper: [raw] 로컬 RawDocument 8개 로드
=== 2. 파싱 ===
INFO parser: 총 850개 청크 파싱 완료
=== 3. 적재 ===
INFO ingest: Qdrant 사용: https://...qdrant.tech / collection=compliance_agents
INFO ingest: VectorStoreIndex 구성 완료
완료: 850개 청크 적재됨
```

Ingest takes **2–5 minutes** on first run (embedding 850 chunks via Ollama). Qdrant Cloud persists the collection so you only need to re-run ingest if the source data changes.

---

## Step 5 — Run a query

```bash
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"
```

Expected output shape:
```
[판정] 조건부 가능
[근거] 표준투자권유준칙 제14조에 따라 고령투자자에게는 별도의 적합성 확인 절차가 필요합니다...
[인용 조항] [{"source_name": "표준투자권유준칙", "citation_id": "제14조"}, ...]
[위험도] 3/3
[팩트체크] 통과 ✓
[실행 에이전트] ['규정', '법규']
[총 토큰 사용량] 2840
```

> **Note**: `query` mode re-ingests on every run (known limitation — index is not persisted between runs).  
> If Qdrant Cloud is configured, re-ingestion is fast after the first run because the collection already holds the vectors.

---

## Demo queries

```bash
# Suitability principle — 적합성원칙
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"

# Explanation duty — 설명의무
python main.py query "ELS 상품의 손실 구조를 고객에게 설명하지 않으면 어떤 제재를 받나요?"

# Case law — 분쟁사례
python main.py query "설명의무 위반으로 손해배상 책임이 인정된 판례가 있나요?"

# Internal controls — 내부통제
python main.py query "준법감시인의 선임 요건과 직무 범위는 무엇인가요?"
```

---

## Observability — Langfuse

After a `query` run, open [cloud.langfuse.com](https://cloud.langfuse.com):

- **Traces tab**: one root trace `compliance_query` per run
  - Nested spans: `classify_step` → `search_규정` / `search_법규` / `search_사례` → `synthesize_step` → `factcheck_step`
  - Each span has structured `input` and `output` fields
- **Prompts tab**: `synthesize_agent` and `factcheck_agent` are auto-uploaded on first query run
  - Edit prompts directly in the Langfuse UI — changes take effect on the next run without any code change

---

## Optional — Add FSS dispute case PDFs

The system supports optional PDF input from the Financial Supervisory Service:

```bash
python main.py ingest ./분쟁사례.pdf
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` on Ollama | Ollama not running | Run `ollama serve` in a separate terminal |
| `Unauthorized` from Qdrant | Wrong API key | Check `QDRANT_API_KEY` in `.env` |
| `Langfuse` warnings in logs | Langfuse not configured | Set `LANGFUSE_*` env vars in `.env`, or ignore (workflow still runs) |
| `사용자 정보 검증에 실패하였습니다` | DRF API key invalid | Only matters for re-scraping; local `data/raw/` files are used by default |
| Empty search results | Qdrant collection empty | Run `python main.py ingest` first |
| Slow first query | Embedding 850 chunks | Expected — subsequent queries are faster |

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model name |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `compliance_agents` | Qdrant collection name |
| `QDRANT_API_KEY` | *(empty)* | Qdrant Cloud API key (empty = local, no auth) |
| `LANGFUSE_PUBLIC_KEY` | *(none)* | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | *(none)* | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server URL |
| `TOKEN_BUDGET` | `32000` | Max total tokens per workflow run |
| `STEP_TOKEN_LIMIT` | `4000` | Max tokens per individual step |
