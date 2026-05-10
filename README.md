# compliance-agents

증권사 임직원이 컴플라이언스 질문을 입력하면, 사규·법규·분쟁사례를 병렬 검색하여 판정(가능/불가/조건부 가능), 인용 조항, 위험도, 팩트체크 결과를 구조화된 형태로 반환하는 멀티에이전트 시스템입니다.

포트폴리오 프로젝트입니다. 설계 의도의 명확성을 우선으로 합니다.

---

## 워크플로우 구조

```
StartEvent(query)
  → [Step 1] classify_step       LLM 라우팅 (constrained JSON) → 활성화 레인 결정
  → [Step 2a] search_규정        HyDE 변환 → 사규 벡터 검색 → citation_id exact-match 검증
  → [Step 2b] search_법규        HyDE 변환 → 법규 벡터 검색 → citation_id exact-match 검증  ← 병렬
  → [Step 2c] search_사례        HyDE 변환 → 분쟝사례 벡터 검색 → 사건번호 metadata 검증
  → [Step 3]  synthesize_step    검증된 근거만 종합 → 판정 JSON 생성
  → [Step 4]  factcheck_step     최종 인용 citation_id 존재 여부 exact-match 재검증
  → StopEvent(result=FinalAnswer)
```

검색 Step은 LLM 요약을 만들지 않고 검증된 근거 객체를 반환합니다.
최종 LLM은 중간 요약이 아니라 검증된 원문 snippet과 metadata만 보고 판정합니다.

LLM이 어떤 검색 함수를 호출할지는 코드가 결정합니다. LLM은 레인 활성화(규정/법규/사례)만 결정하며,
source_name·citation_id 필터는 코드(regex)가 생성합니다. 금융 컴플라이언스 도메인에서 LLM이 하드 필터를 생성하면 잘못된 추론이 무음 zero-recall로 이어질 수 있기 때문입니다.

### 에이전트 활성화 규칙

`classify_step`이 `classify_agent.txt` 프롬프트를 사용해 LLM에게 라우팅을 위임합니다. LLM은 질문의 의미를 분석해 `["규정", "법규", "사례"]` 중 활성화할 레인을 JSON으로 반환하며, 근거(`reasoning`)도 함께 출력합니다.

| 에이전트 | 커버 범위 |
|---|---|
| 규정 | 투자권유, 적합성원칙, 설명의무, 고령·장애인 투자자 보호, 파생상품 권유 제한, 준법감시, 내부통제 |
| 법규 | 법적 의무와 금지행위, 인허가, 과태료·형사처벌, 불공정거래, 공시 의무 |
| 사례 | 실제 소송/분쟝 결과, 손해배상 판결, 법원의 의무 해석 기준 |

LLM이 유효한 에이전트를 반환하지 못하면 세 레인을 모두 활성화하는 fallback이 동작합니다. Langfuse 트레이스의 `classify_step` 스팬에서 LLM의 라우팅 근거를 확인할 수 있습니다.

---

## 데이터 소스

### 사규 — 한국금융투자협회 (KOFIA)

**출처**: `law.kofia.or.kr` (금융투자협회 규정집 포털)  
**수집 방법**: `data/raw`의 저장 HTML을 우선 사용. 로컬 원본이 없으면 HTTP GET 수집.  
**파싱**: BeautifulSoup `div.JO` DOM 구조 기반 파싱. 각 조항·섹션이 `<div class="JO">` 단위로 분리되어 있어 문서 전체 regex 분할을 사용하지 않음.

| 문서명 | 내용 | citation 단위 |
|--------|------|--------|
| **표준투자권유준칙** | 금융투자회사가 투자권유 시 따라야 할 업계 표준 절차. 적합성원칙, 설명의무, 고령투자자·장애인 보호, 파생상품 권유 제한 등을 규정 | `제N조` 조항 |
| **금융투자회사표준내부통제기준** | 준법감시인 업무, 정보차단벽(Chinese Wall), 임직원 금융투자상품 거래 제한, 내부제보 절차 등 내부통제 기준 | `1.1`, `2.2.1` 섹션 |

### 법규 — 법제처 DRF Open API

**출처**: `www.law.go.kr` — 법제처(Ministry of Government Legislation) 공공데이터 Open API  
**API 엔드포인트**: `https://www.law.go.kr/DRF/lawService.do`  
**수집 방법**: `data/raw`의 저장 XML을 우선 사용. 재수집 시 DRF XML API 사용. **개인 OC 키 발급 및 서버 IP 등록 필요** (law.go.kr 공공데이터 포털에서 신청).  
**파싱**: `xml.etree.ElementTree`로 `<조문단위>` 단위 분할. 실제 DRF 구조의 `<항>`, `<호>`, `<목>`을 포함해 본문을 구성하고 `조문가지번호`로 `제N조의M`을 보존

| 문서명 | 내용 | 법령일련번호(MST) | 공포일자 | 조항 수 |
|--------|------|----------------|---------|--------|
| **자본시장과 금융투자업에 관한 법률** (자본시장법) | 금융투자업 인·허가, 투자자 보호, 불공정거래 규제, 집합투자·신탁·파생상품 규율 등 자본시장 전반을 규율하는 기본법 | 273695 | 2025-09-16 (시행 2026-03-17) | 580 |

### 분쟁사례 — 법제처 DRF 판례 + 금융감독원 PDF

#### 법원 판례 (수집 완료)

**출처**: `www.law.go.kr` — 법제처 DRF Open API (`target=prec`)  
**수집 방법**: 판례정보일련번호(ID)로 개별 판례 XML 조회. OC 키 + IP 등록 필요.  
**파싱**: `<판시사항>`, `<판결요지>`, `<판례내용>` 추출; HTML 태그 제거. 판례 1건 = 청크 1개.

| ID | 사건번호 | 법원 | 선고일 | 주요 쟁점 |
|----|---------|------|-------|---------|
| 182205 | 2013나2021183-1 | 서울고등법원 | 2015-04-23 | 회사채 권유 시 설명의무 위반 |
| 204882 | 2016다35352 | 대법원 | 2018-07-20 | 설명의무 범위 및 판단 기준 |
| 204194 | 2015다69853 | 대법원 | 2018-09-28 | 자본시장법 제178조 부정행위 |
| 177551 | 2013다217498 | 대법원 | 2015-01-29 | 금융투자상품 소개와 투자권유 구별 |
| 231803 | 2018도4413 | 대법원 | 2022-10-27 | 투자자문업자 적합성원칙·설명의무 |

판례 ID 목록은 `data/scraper.py`의 `PREC_IDS`에서 관리합니다. 항목 추가 시 이 리스트에 ID만 추가하면 됩니다.

#### 금융감독원 분쟁사례 PDF (선택)

**출처**: 금융감독원 발행 금융분쟁사례집 PDF (사용자가 직접 준비)

```bash
python main.py ingest ./분쟁사례.pdf
```

### 수집 결과 요약

| 구분 | 문서/건 수 | 청크 수 | exact lookup 항목 수 |
|------|----------|--------|----------------------|
| 사규 | 2 | 265 | 265 |
| 법규 | 1 | 580 | 580 |
| 분쟁사례 (판례) | 5 | 5 | 5 |
| **합계** | **8** | **850** | **850** |

원본 HTML/XML은 `data/raw/`에, 판례 XML은 `data/raw/prec/`에 저장됩니다. 데모 실행은 로컬 raw 파일을 먼저 사용합니다.

---

## 설치 및 실행

### 사전 요구 사항

```bash
# Ollama 설치 후 모델 준비
ollama serve                        # 별도 터미널에서 실행
ollama pull qwen2.5:7b              # LLM
ollama pull qwen3-embedding:0.6b    # 임베딩 모델

# Qdrant 실행 (권장)
docker run -p 6333:6333 qdrant/qdrant
```

### 패키지 설치

```bash
uv sync
# 또는
pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama \
            llama-index-vector-stores-qdrant qdrant-client \
            requests beautifulsoup4 pdfminer.six
```

Qdrant가 없거나 `USE_QDRANT=0`이면 LlamaIndex 인메모리 VectorStore로 fallback합니다.

### 실행

```bash
# 데이터 수집 + 인덱스 구성 (Ollama 필요)
python main.py ingest

# PDF 포함 시
python main.py ingest ./분쟁사례.pdf

# 질문 실행
python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"
```

### DRF API 키 설정

법규 데이터 재수집 시 개인 OC 키가 필요합니다.

1. `www.law.go.kr` 공공데이터 포털에서 오픈 API 신청
2. 서버 IP 주소 등록 (IP 미등록 시 인증 오류 발생)
3. `data/scraper.py`의 `DRF_OC = "gamster2"` 를 발급받은 키로 교체

---

## 프로젝트 구조

```
main.py                         # 진입점 (ingest / query 모드)
data/
  scraper.py                    # 웹 수집 → RawDocument
  parser.py                     # RawDocument → ParsedChunk (source-aware citation 단위)
  ingest.py                     # ParsedChunk → Qdrant VectorStoreIndex + article_lookup.json
  raw/                          # 수집된 원본 HTML/XML 캐시
  article_lookup.json           # citation_id exact-match 인덱스 (팩트체크용)
workflow/
  events.py                     # 모든 Event 클래스 (워크플로우 DAG 정의)
  tools.py                      # 검색/조회 함수 4개 + ToolRegistry
  circuit_breaker.py            # 토큰 예산 관리 / 재시도 제어
  compliance_workflow.py        # 5-Step 워크플로우 본문
prompts/
  synthesize_agent.txt
  factcheck_agent.txt
```

**의존 방향**: scraper → parser → ingest → tools → compliance_workflow → main

---

## RAG 구성

- **LLM**: Ollama `qwen2.5:7b`
- **Embedding**: Ollama `qwen3-embedding:0.6b` (`EMBEDDING_MODEL` 환경변수로 교체 가능)
- **Reranker**: `Qwen/Qwen3-Reranker-0.6B` cross-encoder (`USE_RERANKER=0`으로 비활성화 가능)
- **Query transform**: HyDE — 질문을 가상 법령 조항으로 변환 후 임베딩 (Gao et al. 2022)
- **Routing**: `classify_agent.txt` — LLM이 의미 기반으로 레인 활성화 결정 (constrained JSON)
- **Vector DB**: Qdrant Cloud (`QDRANT_URL`, `QDRANT_API_KEY` 환경변수로 설정)
- **Fallback**: Qdrant 초기화 실패 시 인메모리 VectorStore
- **Factcheck**: `article_lookup.json` exact-match dictionary

각 청크는 본문 `text`를 임베딩하고 다음 metadata를 저장합니다.

| 필드 | 목적 |
|---|---|
| `source_type` | 필수 lane filter (`사규`, `법규`, `분쟁사례`) |
| `source_name` | 특정 규정집/법령 precision filter 및 인용 |
| `citation_id` | exact-match 검증의 표준 키 (`제48조`, `2.2.1`, 사건번호) |
| `article_no` | 조항형 문서의 조항번호 |
| `article_title` | 조항 제목 |
| `section_no` | 섹션형 문서의 섹션번호 |
| `section_title` | 섹션 제목 |
| `case_no` | 판례 인용 및 사례 검증 |
| `category` | 고신뢰 주제 precision filter |
| `chunk_id` | 검색 근거 식별자 |
| `verified` | 검증된 근거 여부 |
| `url` | 원문 출처 |

검색은 항상 `source_type` metadata filter를 적용한 뒤 semantic search를 수행합니다.
질문에 `자본시장법`, `표준투자권유준칙`, `제48조`, `2.2.1`처럼 명시적 문서명·조항번호가 있으면 `source_name`, `citation_id` precision filter를 추가합니다. `category` 분류는 키워드 오탐 가능성이 있어 AND 조건에서 제외합니다.

---

## 출력 형태

```
[판정] 조건부 가능
[근거] 표준투자권유준칙 제14조에 따라 고령투자자(65세 이상)에게는 별도의 적합성 확인 절차가
       필요합니다. 레버리지 ETF는 고위험 상품으로, 투자자의 위험 감수 능력 확인 및 충분한
       설명의무 이행이 선행되어야 합니다.
[인용 조항] [{"source_name": "표준투자권유준칙", "citation_id": "제14조"}, ...]
[위험도] 3/3
[팩트체크] 통과 ✓
[실행 에이전트] ['search_규정', 'search_법규', 'synthesize', 'factcheck']
[총 토큰 사용량] 2840
```

---

## 한계 및 TODO

- **긴 텍스트 chunking**: 현재는 citation 1개/판례 1건을 청크 1개로 유지합니다. 긴 판례와 대형 조항을 더 작은 하위 단위로 나누는 것은 향후 개선 사항입니다.
- **Qdrant 재적재 방식**: `query` 실행 시마다 현재 raw data를 다시 ingest합니다. 데모 단순성을 위한 선택이며, 운영형 구조에서는 persistent collection 재사용 경로가 필요합니다.
- **Ollama 직렬 처리**: Step 2a/b/c는 비동기 병렬 설계이나, Ollama는 요청을 순차 처리합니다. 실제 병렬화는 동시 추론을 지원하는 모델 서버가 필요합니다.
- **토큰 수 추정치**: `circuit_breaker.py`의 토큰 카운트는 상수 추정값입니다. 정확한 카운트는 LLM 응답의 `usage` 필드 파싱이 필요합니다.
