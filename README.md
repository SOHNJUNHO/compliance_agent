# compliance-agents

증권사 임직원이 컴플라이언스 질문을 입력하면, 사규·법규·분쟁사례를 검색하여 근거 기반 답변과 인용 조항을 구조화된 형태로 반환하는 멀티에이전트 시스템입니다. 허용 여부("~ 가능한가?")뿐 아니라 규정·법규·판례를 묻는 정보성 질문에도 답합니다.

증권사 AI 에이전트 개발자 포지션 지원용 포트폴리오 프로젝트이며, 성능 최적화보다 **설계 의도의 명확성**을 우선합니다. 자세한 셋업 방법은 [SETUP.md](SETUP.md)를 참조하세요.

---

## 워크플로우 구조

```
StartEvent(query)
  ├─ [Step 1] classify_step       LLM 라우팅 (constrained JSON) → 활성화 레인 결정
  ├─ [Step 2a] search_규정        HyDE → 사규 벡터 검색 → citation_id exact-match 검증
  ├─ [Step 2b] search_법규        HyDE → 법규 벡터 검색 → citation_id exact-match 검증   ← 병렬 fan-out
  ├─ [Step 2c] search_사례        HyDE → 분쟁사례 벡터 검색 → 사건번호 metadata 검증
  └─ [Step 3]  synthesize_step    검증된 근거만 합성 → 근거 기반 답변 JSON → StopEvent(FinalAnswer)
```

LlamaIndex `Workflow`의 `@step` 어노테이션 타입이 곧 라우팅 규칙입니다. 검색 step은 LLM 요약을 만들지 않고 검증된 근거 객체만 반환하며, 최종 LLM은 검증된 원문 snippet과 metadata만 보고 답변합니다.

### Rule-based ↔ LLM 권한 분리

LLM은 레인 활성화(`["규정","법규","사례"]` 중 선택)만 결정합니다. `source_name`·`citation_id` 정밀 필터는 코드(regex)가 만듭니다. 하드 AND 필터에 LLM의 잘못된 추론이 들어가면 **무음 zero-recall**로 이어지기 때문입니다.

| 결정 주체 | 영역 | 실패 영향 |
|---|---|---|
| LLM | 레인 활성화 | 잘못 추론하면 "관련 없는 레인을 함께 검색" — 복구 가능 |
| 코드 | precision 필터 (`source_name` / `citation_id`) | 잘못 추론하면 "관련 청크가 0건 반환" — 복구 불가 |
| 코드 | source_type 필터 | `regulation_search` ↔ 사규 / `law_search` ↔ 법규 / `case_search` ↔ 분쟁사례 (하드코딩) |

LLM이 유효한 레인을 반환하지 못하면 세 레인을 모두 활성화하는 안전 fallback이 동작합니다.

---

## 데이터 소스

| 구분 | source_name | 수집 방법 | 청크 단위 | 청크 수 |
|------|-------------|---------|---------|--------|
| 사규 | 표준투자권유준칙 | KOFIA HTML — `div.JO` DOM 단위 | `제N조` 조항 | 47 |
| 사규 | 금융투자회사표준내부통제기준 | KOFIA HTML — `div.JO` DOM 단위 | `1.1` / `2.2.1` 섹션 | 218 |
| 법규 | 자본시장과 금융투자업에 관한 법률 (자본시장법) | 법제처 DRF Open API (target=law, MST=273695) | `<조문단위>` | 580 |
| 분쟁사례 | 법원판례 (5건) | 법제처 DRF Open API (target=prec) | 판례 1건 | 5 |
| **합계** |  |  |  | **850** |

원본 HTML/XML은 `data/raw/`에 캐시되며, `scrape_all()`은 로컬 캐시가 있으면 웹 재요청 없이 그대로 사용합니다. 판례 ID 목록은 `data/scraper.py`의 `PREC_IDS`에서 관리합니다.

법규 / 판례 재수집 시 법제처 DRF API의 OC 키와 IP 등록이 필요합니다. 인증 실패 시 XML 응답에 `사용자 정보 검증에 실패하였습니다.`가 들어옵니다.

---

## 데이터 흐름

```
scraper.py    →  RawDocument         (HTML / XML / PDF 원본)
parser.py     →  ParsedChunk          (citation 1개 = 청크 1개, source-aware)
ingest.py     →  Qdrant VectorStoreIndex   (벡터 검색용)
```

별도의 `article_lookup.json` 파일은 없습니다. query 시작 시 `load_lookup_table()`이 Qdrant 컬렉션을 1회 scroll하여 `source_name||citation_id → 조항 dict` 인메모리 테이블을 구성합니다. 벡터 검색은 "비슷한" 문서를 찾고, lookup은 "정확히 존재하는지"를 확인합니다. 두 조회는 역할이 분리되어 있습니다.

Ingest와 query는 완전히 분리되어 있습니다. `ingest`는 한 번만 실행하고 (`USE_QDRANT=1`이면 Qdrant Cloud에 영속), `query`는 기존 컬렉션을 재임베딩 없이 로드합니다.

---

## 실행

자세한 환경설정은 [SETUP.md](SETUP.md)를 참조하세요. 핵심만:

```bash
# 1. Ollama 실행 (별도 터미널)
ollama serve
ollama pull qwen3:8b-q4_K_M           # LLM (~5.2 GB)
ollama pull qwen3-embedding:0.6b-q8_0 # embedding (~0.6 GB)

# 2. 환경변수
cp .env.example .env                  # Qdrant Cloud / Langfuse 키 입력

# 3. 의존성
uv sync

# 4. 1회 ingest (Qdrant에 영속됨)
python run_ingest.py

# 5. 반복 query
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"
```

### 하드웨어 / 실행 시간 (Apple M5 기준)

| 단계 | 소요 시간 |
|------|---------|
| `ingest` (850청크 임베딩 + Qdrant 업로드) | ~4–6분 |
| `query` (HyDE 3회 + 검색 + reranker + 합성) | ~2분 |
| Reranker 첫 로드 (HuggingFace 다운로드) | +1회 ~1.5GB |

병렬로 표시되는 Step 2a/b/c는 실제로는 Ollama가 요청을 직렬 처리하므로 wall-clock은 직렬에 가깝습니다.

---

## 출력 예시

```
[답변]
표준투자권유준칙||제14조
  → 고령투자자(65세 이상)에게는 별도의 적합성 확인 절차가 필요합니다.
자본시장과금융투자업에관한법률||제46조
  → 적합성 원칙에 따라 투자자 성향을 먼저 확인해야 합니다.
[사용된 근거 ID] ['표준투자권유준칙||제14조', '자본시장과금융투자업에관한법률||제46조']
[실행 에이전트] ['규정', '법규']
[활성화 근거] 고령투자자 투자권유는 사규의 적합성원칙과 법적 의무 모두 관련됩니다.

[인용 근거]
표준투자권유준칙||제14조
  "제14조(고령투자자 보호) 회사는 65세 이상 ..."
```

---

## 프로젝트 구조

```
main.py                   # 진입점 (query 전용; 데이터 적재는 run_ingest.py)
data/
  scraper.py              # 로컬 raw 우선, 없으면 웹 fallback → RawDocument
  parser.py               # RawDocument → ParsedChunk (source-aware citation 단위)
  ingest.py               # ParsedChunk → Qdrant (lookup 테이블은 query 시 scroll로 구성)
  raw/                    # 수집된 HTML/XML 캐시 (재현성)
  raw/prec/               # 판례 XML
workflow/
  events.py               # 모든 Event 클래스 (Workflow DAG 정의)
  tools.py                # 4개 검색/조회 함수 + ToolRegistry
  evidence.py             # 근거 추출·검증·포맷 순수 함수
  compliance_workflow.py  # 5-Step Workflow 본문
  reranker.py             # BAAI/bge-reranker-v2-m3 cross-encoder (USE_RERANKER=0 비활성화)
  langfuse_setup.py       # Langfuse 클라이언트 + 프롬프트 관리
prompts/
  classify_agent.txt      # 레인 라우팅
  hyde_regulation.txt     # 쿼리 → 가상 사규 조항 변환 (HyDE)
  hyde_law.txt            # 쿼리 → 가상 법령 조항 변환 (HyDE)
  hyde_case.txt           # 쿼리 → 가상 판례 판시사항 변환 (HyDE)
  synthesize_agent.txt    # per-passage 합성 LLM 시스템 프롬프트
```

**의존 방향**: `scraper → parser → ingest → tools → compliance_workflow → main`

---

## RAG 구성

- **LLM**: Ollama `qwen3:8b-q4_K_M` (json_mode=True)
- **Embedding**: Ollama `qwen3-embedding:0.6b-q8_0` (1024-dim)
- **Vector DB**: Qdrant Cloud, TurboQuantization BITS4 (always_ram=True), payload index 5개 필드
- **Reranker**: `BAAI/bge-reranker-v2-m3` cross-encoder (HuggingFace 자동 다운로드 ~2.2 GB)
- **Query transform**: HyDE — 질문을 레인별 코퍼스 문체의 가상 문서로 변환 후 임베딩 (Gao et al. 2022)
- **Routing**: `classify_agent.txt` — LLM이 의미 기반으로 레인 활성화 결정
- **검증(Validation)**: Qdrant scroll 기반 인메모리 테이블 (`source_name||citation_id → dict`)로 검색 단계에서 citation_id 존재를 확인 (query 시작 시 1회 로드)
- **Observability**: Langfuse — `@observe` 수동 계측으로 6개 스팬(루트 compliance_query + classify + search×3 + synthesize)을 기록한다. LlamaIndex 자동 트레이싱도 활성화되어 있어 raw LLM/embedding 스팬이 추가로 기록된다. Langfuse 미설정 시 모든 호출이 no-op fallback이 되어 워크플로우 실행에는 영향 없다.

각 청크는 본문 `text`를 임베딩하고 다음 metadata를 저장합니다.

| 필드 | 목적 |
|---|---|
| `source_type` | lane filter (`사규`, `법규`, `분쟁사례`) — tools.py에 하드코딩 |
| `source_name` | 규정집/법령 이름 — 답변 인용 표시에 사용 |
| `citation_id` | exact-match 검증의 표준 키 (`제48조`, `2.2.1`, 사건번호) |
| `article_no` | 조항번호 (조항형 문서 전용) |
| `section_no` | 섹션번호 (섹션형 문서 전용) |
| `case_no` | 사건번호 (분쟁사례 전용) |
| `url` | 원문 출처 (워크플로우 미사용) |

---

## 한계 및 TODO

- **LLM 호출 장애 처리는 최소 수준**: `_structured_predict_with_repair`가 ValidationError는 최대 1회 자가 수정, 전송 오류(httpx·ollama)는 최대 2회 지수 백오프 재시도를 수행합니다. per-step deadline / degraded fallback 응답 / circuit breaker는 구현되지 않았습니다. 운영 환경에서는 추가 정교화가 필요합니다.
- **Ollama 직렬 처리**: Step 2a/b/c는 비동기 병렬 설계이지만 Ollama가 요청을 순차 처리하므로 실제 wall-clock 병렬화는 안 됩니다. 동시 추론이 가능한 모델 서버로 LLM을 교체해야 합니다.
- **토큰 사용량 추적은 미구현**: LLM 응답의 `usage` 필드 파싱 및 per-run 토큰 집계는 구현되지 않았습니다. 운영 환경에서는 Langfuse 트레이스의 `usage` 스팬 또는 별도 계측으로 추가할 수 있습니다.
- **PDF 파서는 best-effort**: 금감원 분쟁사례 PDF의 사례 경계 regex는 표준 형식 가정에 의존합니다. 다른 PDF는 별도 검증이 필요합니다.
- **법제처 DRF API 의존**: 법규/판례 재수집은 OC 키 + 등록된 IP가 필요합니다. 평소 실행은 `data/raw/` 캐시로 충당됩니다.
- **Query 결과는 stdout만**: 실행 이력이 필요하면 Langfuse 트레이스를 활용하거나 `main.py`에 파일/DB 출력을 추가하세요.
