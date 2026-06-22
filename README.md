# compliance-agents

증권사 임직원이 컴플라이언스 질문을 입력하면, 사규·법규·분쟁사례를 검색하여 근거 기반 답변과 인용 조항을 구조화된 형태로 반환하는 멀티에이전트 시스템입니다. 허용 여부("~ 가능한가?")뿐 아니라 규정·법규·판례를 묻는 정보성 질문에도 답합니다.

> **이 브랜치(`feature/langgraph-port`)는 LangGraph + LangChain 포팅 버전입니다.**
> 원본 LlamaIndex 기반 구현은 `main` 브랜치를 참조하세요.

---

## 이 브랜치에서 바뀐 것

### 핵심 개념 차이

LlamaIndex `Workflow`는 **이벤트 구동(event-driven)** 방식입니다. `@step` 메서드가 `Event` 서브클래스를 받아 반환하고, LlamaIndex가 타입 어노테이션으로 DAG를 자동 구성합니다. Step 간 공유 상태는 `ctx.store`에, fan-in 동기화는 `ctx.collect_events()`에 의존합니다.

LangGraph는 **상태 기반(state-based)** 방식입니다. 단일 `TypedDict`(`ComplianceState`)가 노드 간을 흐르고, 노드는 상태의 부분 딕셔너리를 반환합니다. reducer가 병합을 담당하고, BSP(Bulk Synchronous Parallel) 슈퍼스텝 실행 모델이 fan-in을 자연스럽게 처리합니다.

### 컴포넌트 대응표

| LlamaIndex (`main`) | LangGraph/LangChain (`feature/langgraph-port`) |
|---|---|
| `Workflow` + `@step` 메서드 | `StateGraph` + 노드 함수 |
| `Event` 클래스 5종 | 단일 `ComplianceState` TypedDict |
| `ctx.collect_events()` fan-in | `Annotated[list, operator.add]` reducer 자동 병합 |
| `skipped=True` 센티널 + 3개 Step 래퍼 | `Send` API 동적 fan-out (활성 레인만 파견) |
| `ctx.store.set/get(...)` | 상태 필드 직접 읽기/쓰기 |
| `VectorStoreIndex` + `MetadataFilters` | `QdrantVectorStore` + native qdrant `Filter` |
| 필터 키: `source_type` | 필터 키: **`metadata.source_type`** (langchain_qdrant 중첩 구조) |
| `OllamaEmbedding` | `OllamaEmbeddings` (langchain-ollama) |
| `Ollama(json_mode=True)` | `ChatOllama` (langchain-ollama) |
| `llm.astructured_predict(Model)` | `llm.with_structured_output(Model, method="json_schema")` |
| `SentenceTransformerRerank` | `CrossEncoderReranker` + `HuggingFaceCrossEncoder` (langchain) |
| `TextNode(text, metadata)` | `Document(page_content, metadata)` |
| `@observe` + `LlamaIndexInstrumentor` | `langfuse.langchain.CallbackHandler` (`graph.ainvoke` config에 전달) |
| Qdrant 컬렉션: `compliance_agent` | Qdrant 컬렉션: **`compliance_agent_lc`** (페이로드 호환 불가로 분리) |

### 변경되지 않은 것

`data/scraper.py`, `data/parser.py`, `data/raw/`, `prompts/*.txt`, `manage_prompts.py`, `workflow/langfuse_setup.py`, `workflow/evidence.py`(순수 함수)는 그대로 재사용됩니다. LLM 모델(`qwen3:8b-q4_K_M`), 임베딩 모델(`qwen3-embedding:0.6b`), Qdrant 설정(TurboQuantization BITS4), reranker 모델(`BAAI/bge-reranker-v2-m3`)도 동일합니다.

---

## 워크플로우 구조

```
START
  → classify              LLM 라우팅 → agent_list, routing_reasoning
  → route_to_lanes        Send ×N 반환 (활성 레인 수만큼 search 태스크 파견)
  → search ×N             병렬 실행: HyDE → 검색+재순위 → 검증 → evidence 누적(reducer)
  → synthesize            정렬(규정→법규→사례) → per-passage LLM map → 코드 reduce → 최종 답변
  → END
```

`route_to_lanes`는 `agent_list`에 있는 레인마다 `Send("search", LaneWork)`를 반환합니다. LlamaIndex 버전에서 비활성 레인이 `skipped=True` 이벤트를 반환하던 패턴은 사라졌습니다 — 파견되지 않은 레인은 아예 실행되지 않습니다.

`search` 노드가 반환하는 `{"evidence": [...]}` 는 `operator.add` reducer가 자동으로 누적합니다. 같은 슈퍼스텝의 모든 `search` 태스크가 완료된 후 `synthesize`가 실행됩니다(BSP 보장).

### Rule-based ↔ LLM 권한 분리

LLM은 레인 활성화(`["규정","법규","사례"]` 중 선택)만 결정합니다. `source_type` 필터는 `tools_lc.py`에서 레인별로 하드코딩됩니다. 잘못된 LLM 추론이 필터로 전파되면 **무음 zero-recall**이 발생하기 때문입니다.

| 결정 주체 | 영역 | 실패 영향 |
|---|---|---|
| LLM | 레인 활성화 | 잘못 추론하면 "관련 없는 레인도 함께 검색" — 복구 가능 |
| 코드 | `source_type` 필터 (`metadata.source_type`) | `regulation_search` ↔ 사규 / `law_search` ↔ 법규 / `case_search` ↔ 분쟁사례 (하드코딩) |

LLM이 유효한 레인을 반환하지 못하면 세 레인을 모두 활성화하는 안전 fallback이 동작합니다.

---

## 데이터 적재 (중요 변경)

### 왜 별도 컬렉션인가

LlamaIndex와 LangChain은 Qdrant 페이로드 구조가 호환되지 않습니다.

| 구현 | Qdrant 페이로드 구조 |
|---|---|
| LlamaIndex (`main`) | `{"text": "...", "source_type": "사규", ...}` (flat) |
| LangChain (이 브랜치) | `{"page_content": "...", "metadata": {"source_type": "사규", ...}}` (nested) |

기존 `compliance_agent` 컬렉션을 그대로 가리키면 `metadata.source_type` 필터가 일치하는 문서를 찾지 못해 검색이 무음으로 실패합니다. 따라서 **`compliance_agent_lc`** 컬렉션에 별도 적재합니다. `compliance_agent` 컬렉션은 변경되지 않으므로 `main` 브랜치는 그대로 동작합니다.

페이로드 인덱스도 중첩 키 기준으로 생성됩니다: `metadata.source_type`, `metadata.source_name`, `metadata.citation_id`, `metadata.article_no`, `metadata.case_no`.

### 적재 명령

```bash
python data/run_ingest_lc.py               # 웹 데이터만 수집
python data/run_ingest_lc.py ./fss.pdf     # PDF 추가 적재
```

컬렉션 이름은 `QDRANT_COLLECTION_LC` 환경변수로 오버라이드할 수 있습니다 (기본값 `compliance_agent_lc`).

---

## 실행

```bash
# 1. Ollama 실행 (별도 터미널)
ollama serve
ollama pull qwen3:8b-q4_K_M              # LLM (~5.2 GB)
ollama pull qwen3-embedding:0.6b-q8_0   # embedding (~0.6 GB)

# 2. 환경변수
cp .env.example .env                     # Qdrant Cloud / Langfuse 키 입력

# 3. 의존성
uv sync

# 4. 1회 ingest (compliance_agent_lc 컬렉션 생성)
python data/run_ingest_lc.py

# 5. 반복 query
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?" user-001
```

### 하드웨어 / 실행 시간 (Apple M5 기준)

| 단계 | 소요 시간 |
|------|---------|
| `ingest` (850청크 임베딩 + Qdrant 업로드) | ~4–6분 |
| `query` (HyDE 3회 + 검색 + reranker + 합성) | ~2분 |
| Reranker 첫 로드 (HuggingFace 다운로드) | +1회 ~2.2 GB |

---

## 프로젝트 구조

```
main.py                       # 진입점 (ChatOllama + build_graph + Langfuse CallbackHandler)
data/
  scraper.py                  # 변경 없음 — 로컬 raw 우선, 없으면 웹 fallback
  parser.py                   # 변경 없음 — RawDocument → ParsedChunk
  ingest_lc.py                # [신규] ParsedChunk → Document → Qdrant (compliance_agent_lc)
  run_ingest_lc.py            # [신규] LangChain 적재 파이프라인 진입점
  ingest.py                   # LlamaIndex 버전 유지 (parity 확인 후 제거 예정)
  raw/                        # 수집된 HTML/XML 캐시 (공유)
workflow/
  state.py                    # [신규] ComplianceState TypedDict + LaneWork + LLM 계약
  graph.py                    # [신규] StateGraph 본문 (classify/search/synthesize 노드)
  tools_lc.py                 # [신규] QdrantVectorStore 기반 검색 함수 + ToolRegistry
  reranker_lc.py              # [신규] CrossEncoderReranker + HuggingFaceCrossEncoder
  evidence.py                 # 변경 없음 — 근거 추출·검증·포맷 순수 함수
  langfuse_setup.py           # 변경 없음 — Langfuse 클라이언트 + 프롬프트 관리
  compliance_workflow.py      # LlamaIndex 버전 유지 (parity 확인 후 제거 예정)
  events.py                   # LlamaIndex 버전 유지 (parity 확인 후 제거 예정)
  tools.py                    # LlamaIndex 버전 유지 (parity 확인 후 제거 예정)
  reranker.py                 # LlamaIndex 버전 유지 (parity 확인 후 제거 예정)
prompts/                      # 변경 없음 — 5개 프롬프트 파일 공유
```

**의존 방향**: `scraper → parser → ingest_lc → tools_lc → graph → main`

---

## 데이터 소스

변경 없음. `main` 브랜치 README 참조.

| 구분 | source_name | 청크 수 |
|------|-------------|--------|
| 사규 | 표준투자권유준칙 | 47 |
| 사규 | 금융투자회사표준내부통제기준 | 218 |
| 법규 | 자본시장과 금융투자업에 관한 법률 | 580 |
| 분쟁사례 | 법원판례 (5건) | 5 |
| **합계** | | **850** |

---

## RAG 구성

- **LLM**: `ChatOllama(model="qwen3:8b-q4_K_M", temperature=0)`
- **Embedding**: `OllamaEmbeddings(model="qwen3-embedding:0.6b")` (1024-dim, 클린 텍스트 임베딩)
- **Vector DB**: Qdrant Cloud, 컬렉션 `compliance_agent_lc`, TurboQuantization BITS4 (always_ram=True)
- **Reranker**: `CrossEncoderReranker(HuggingFaceCrossEncoder("BAAI/bge-reranker-v2-m3"))` — 레인별 top_n 3/3/2
- **Query transform**: HyDE — 질문을 레인별 코퍼스 문체의 가상 문서로 변환 후 임베딩 (Gao et al. 2022)
- **Routing**: `classify_agent.txt` — LLM이 의미 기반으로 레인 활성화 결정, ValidationError 시 전체 레인 fallback
- **구조화 출력**: `with_structured_output(Model, method="json_schema")` — `ClassifyResponse` / `PassageAnswer` Pydantic 스키마로 LLM 출력을 제약
- **Observability**: `langfuse.langchain.CallbackHandler`를 `graph.ainvoke(config={"callbacks": [handler]})` 에 전달. 모든 노드·LLM 호출·리트리버 호출이 자동으로 Langfuse 트레이스에 기록된다. `main` 브랜치의 `@observe` 수동 계측과 `LlamaIndexInstrumentor` 자동 트레이싱을 단일 handler로 대체. Langfuse 미설정 시 워크플로우 실행에 영향 없음.

각 청크는 `Document.page_content`를 임베딩하고 다음 metadata를 저장합니다.

| 필드 | 목적 |
|---|---|
| `source_type` | lane filter (`사규`, `법규`, `분쟁사례`) — tools_lc.py에 하드코딩 |
| `source_name` | 규정집/법령 이름 — 답변 인용 표시에 사용 |
| `citation_id` | 검증의 표준 키 (`제48조`, `2.2.1`, 사건번호) |
| `article_no` | 조항번호 (조항형 문서 전용) |
| `section_no` | 섹션번호 (섹션형 문서 전용) |
| `case_no` | 사건번호 (분쟁사례 전용) |
| `url` | 원문 출처 (워크플로우 미사용) |

---

## 출력 예시

```
==================================================
질문: 65세 일반투자자에게 원금손실 가능성이 큰 레버리지 ETF를 권유할 때 주의해야 할 점이 무엇인가요?
==================================================

[답변]
표준투자권유준칙||제14조
  → 고령투자자(65세 이상)에게는 별도의 적합성 확인 절차가 필요합니다.
자본시장과금융투자업에관한법률||제46조
  → 적합성 원칙에 따라 투자자 성향을 먼저 확인해야 합니다.

[사용된 근거 ID] ['표준투자권유준칙||제14조', '자본시장과금융투자업에관한법률||제46조']
[실행 에이전트] ['규정', '법규']
[인용 에이전트] ['규정', '법규']
[활성화 근거] 고령투자자 투자권유는 사규의 적합성원칙과 법적 의무 모두 관련됩니다.

[인용 근거]
표준투자권유준칙||제14조
  "제14조(고령투자자 보호) 회사는 65세 이상 ..."
```

---

## 한계 및 TODO

- **LangGraph 버전 parity 미검증**: `compliance_agent_lc` 컬렉션 인제스트 후 `main` 브랜치 결과와 비교 필요. parity 확인 후 LlamaIndex 모듈(`compliance_workflow.py`, `events.py`, `tools.py`, `reranker.py`, `ingest.py`) 및 관련 의존성 제거 예정.
- **임베딩 전략 tradeoff**: LlamaIndex는 `source_type`/`source_name`/`section_no`를 임베딩 텍스트에 자동 주입하지만 이 브랜치는 `page_content`(원문)만 임베딩합니다(클린 텍스트 전략). parity 체크 시 검색 품질이 낮으면 `"{source_name}\n{text}"` 헤더를 추가하고 재인제스트합니다.
- **LLM 호출 장애 처리는 최소 수준**: `_structured_invoke_with_repair`가 ValidationError는 최대 1회 자가 수정, 전송 오류(httpx·ollama)는 최대 2회 지수 백오프 재시도를 수행합니다. per-step deadline / circuit breaker는 구현되지 않았습니다.
- **Ollama 직렬 처리**: `search` 노드는 비동기 병렬 설계이지만 Ollama가 요청을 순차 처리하므로 실제 wall-clock 병렬화는 안 됩니다.
- **토큰 사용량 추적은 미구현**: Langfuse CallbackHandler가 LLM 호출을 자동 기록하므로 Langfuse 대시보드에서 확인 가능하나, 코드 내 집계는 없습니다.
- **PDF 파서는 best-effort**: 금감원 분쟁사례 PDF의 사례 경계 regex는 표준 형식 가정에 의존합니다.
- **법제처 DRF API 의존**: 법규/판례 재수집은 OC 키 + 등록된 IP가 필요합니다. 평소 실행은 `data/raw/` 캐시로 충당됩니다.
