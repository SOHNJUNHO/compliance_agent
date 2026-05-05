# =============================================================================
# scraper.py
# -----------------------------------------------------------------------------
# 역할: 외부 웹사이트에서 원시 텍스트를 수집하여 RawDocument 객체로 반환한다.
#       이 파일은 데이터 파이프라인의 첫 번째 단계이다.
#
# 데이터 흐름:
#   scraper.py → parser.py → ingest.py → Vector DB
#
# 수집 대상:
#   1. law.kofia.or.kr  — 금융투자협회 사규 (표준투자권유준칙 등)
#   2. law.go.kr        — 국가법령정보센터 (자본시장법 등)
#   3. PDF 파일         — 금감원 분쟁사례 (사용자가 직접 준비)
#
# 설계 원칙:
#   - 각 수집 함수는 실패해도 빈 리스트를 반환한다 (워크플로우를 멈추지 않음)
#   - RawDocument는 파싱 전 원시 데이터만 보관한다 (관심사 분리)
#   - REQUEST_DELAY로 서버 부하를 방지한다
# =============================================================================

import time          # REQUEST_DELAY 구현에 사용
import logging       # 수집 과정 로그 출력
import unicodedata

from dataclasses import dataclass, field  # RawDocument 데이터 클래스 정의
from pathlib import Path
from typing import Optional               # Optional 타입 힌트

import requests      # HTTP GET 요청으로 웹페이지 HTML 가져오기

# 로거: __name__ = "scraper" 로 로그 출처를 명확히 표시
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTTP 요청 헤더: 일반 브라우저처럼 보이게 해서 차단을 방지
HEADERS = {"User-Agent": "Mozilla/5.0 (compliance-portfolio-project)"}

# 요청 간 대기 시간 (초): 연속 요청으로 서버 부하를 주지 않기 위함
REQUEST_DELAY = 1.0
RAW_DIR = Path("data/raw")


# =============================================================================
# RawDocument: 수집된 원시 문서를 담는 데이터 클래스
# -----------------------------------------------------------------------------
# @dataclass: __init__, __repr__ 등을 자동 생성
# parser.py의 PARSERS 딕셔너리가 source_type을 키로 파서를 선택한다
# =============================================================================
@dataclass
class RawDocument:
    """파싱 전 원시 문서. parser.py의 입력값."""

    # "사규" | "법규" | "분쟁사례" — parser.py가 이 값으로 파서 함수를 선택
    source_type: str

    # 규정의 공식 이름 (예: "표준투자권유준칙") — 최종 답변 인용 시 표시
    source_name: str

    # 원본 페이지 URL — 사용자가 출처를 직접 확인할 수 있도록 보존
    url: str

    # HTML 원본 문자열 (파싱 전)
    # PDF의 경우 빈 문자열로 설정하고 extra["pdf_path"]를 사용
    raw_html: str

    # 출처별 추가 메타데이터
    # field(default_factory=dict): 인스턴스마다 독립적인 딕셔너리 생성
    # (딕셔너리/리스트를 기본값으로 쓸 때 반드시 field()를 써야 인스턴스 간 공유를 막을 수 있다)
    extra: dict = field(default_factory=dict)


# =============================================================================
# 1. 사규 수집: law.kofia.or.kr
# =============================================================================

# 수집할 kofia 규정 목록
# historySeq: 특정 개정 버전을 고정해서 데이터 재현성을 보장
# 새 규정을 추가하려면 이 리스트에 항목을 추가하면 된다
KOFIA_TARGETS = [
    {
        "source_name": "표준투자권유준칙",
        "url": "https://law.kofia.or.kr/service/law/lawFullScreenContent.do?seq=149&historySeq=428",
    },
    {
        "source_name": "금융투자회사표준내부통제기준",
        "url": "https://law.kofia.or.kr/service/law/lawFullScreenContent.do?seq=150&historySeq=439",
    },
]


def scrape_kofia() -> list[RawDocument]:
    """금융투자협회 규정 페이지 HTML 수집."""
    results = []

    for target in KOFIA_TARGETS:
        try:
            # timeout=10: 10초 안에 응답 없으면 예외 발생
            resp = requests.get(target["url"], headers=HEADERS, timeout=10)
            # HTTP 4xx/5xx 응답이면 예외 발생 (예: 404 Not Found)
            resp.raise_for_status()

            results.append(RawDocument(
                source_type="사규",
                source_name=target["source_name"],
                url=target["url"],
                raw_html=resp.text,   # HTML 원본 그대로 보존
            ))
            logger.info(f"[kofia] 수집 완료: {target['source_name']}")
            time.sleep(REQUEST_DELAY)  # 다음 요청 전 대기

        except requests.RequestException as e:
            # 네트워크 오류, 타임아웃 등 — 이 항목 건너뛰고 계속 진행
            logger.warning(f"[kofia] 수집 실패 ({target['source_name']}): {e}")

    return results


# =============================================================================
# 2. 법규 수집: law.go.kr
# =============================================================================

# 법제처 DRF Open API 설정
# OC: 신청한 사용자 인증키 (law.go.kr 공공데이터 포털에서 발급)
# MST: 법령일련번호 (lawSearch로 조회하여 확인)
DRF_BASE = "https://www.law.go.kr/DRF/lawService.do"
DRF_OC = "gamster2"

LAWGOKR_TARGETS = [
    {
        "source_name": "자본시장과금융투자업에관한법률",
        # MST: lawSearch API로 확인한 현행 자본시장법 법령일련번호
        "mst": "273695",
    },
]


def scrape_lawgokr() -> list[RawDocument]:
    """
    법제처 DRF Open API로 법령 XML 수집.
    raw_html 필드에 XML 문자열을 저장하고, extra["format"]="drf_xml"로 표시.
    """
    results = []

    for target in LAWGOKR_TARGETS:
        url = f"{DRF_BASE}?OC={DRF_OC}&target=law&MST={target['mst']}&type=XML"
        try:
            resp = requests.get(
                DRF_BASE,
                params={"OC": DRF_OC, "target": "law", "MST": target["mst"], "type": "XML"},
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()

            results.append(RawDocument(
                source_type="법규",
                source_name=target["source_name"],
                url=url,
                raw_html=resp.text,
                extra={"format": "drf_xml"},
            ))
            logger.info(f"[law.go.kr] 수집 완료: {target['source_name']}")
            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            logger.warning(f"[law.go.kr] 수집 실패 ({target['source_name']}): {e}")

    return results


# =============================================================================
# 3. 판례 수집: law.go.kr DRF (target=prec)
# =============================================================================

# 수집할 판례 ID 목록 (판례정보일련번호)
# 자본시장법 관련 설명의무·적합성원칙 위반 손해배상 판례
PREC_IDS = [
    "182205",   # 서울고법 2015 — 회사채 권유 설명의무 위반
    "204882",   # 대법원 2018 — 설명의무 범위
    "204194",   # 대법원 2018 — 자본시장법 제178조 부정행위
    "177551",   # 대법원 2015 — 투자권유 적합성원칙
    "231803",   # 대법원 2022 — 투자자문업자 적합성·설명의무
]


def scrape_prec() -> list[RawDocument]:
    """
    법제처 DRF API로 판례 XML 수집 (target=prec).
    각 판례 ID당 하나의 RawDocument를 반환한다.
    extra["format"]="prec_xml"로 표시하여 parser가 분기할 수 있게 한다.
    """
    results = []

    for prec_id in PREC_IDS:
        url = f"{DRF_BASE}?OC={DRF_OC}&target=prec&ID={prec_id}&type=XML"
        try:
            resp = requests.get(
                DRF_BASE,
                params={"OC": DRF_OC, "target": "prec", "ID": prec_id, "type": "XML"},
                headers=HEADERS,
                timeout=20,
            )
            resp.raise_for_status()

            results.append(RawDocument(
                source_type="분쟁사례",
                source_name="법원판례",
                url=url,
                raw_html=resp.text,
                extra={"format": "prec_xml", "prec_id": prec_id},
            ))
            logger.info(f"[prec] 수집 완료: {prec_id}")
            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            logger.warning(f"[prec] 수집 실패 ({prec_id}): {e}")

    return results


# =============================================================================
# 4. 분쟁사례 PDF 로드
# =============================================================================

def load_pdf_as_raw(
    pdf_path: str,
    source_name: str = "금감원분쟁사례"
) -> Optional[RawDocument]:
    """
    사용자가 직접 준비한 PDF를 RawDocument로 래핑한다.
    실제 텍스트 추출은 parser.py의 parse_pdf()에서 수행한다.
    이 함수는 파일 존재 여부만 확인한다 — 역할을 최소화해 관심사 분리를 유지.

    Args:
        pdf_path:    PDF 파일 경로
        source_name: 규정명

    Returns:
        RawDocument (파일 없으면 None)
    """
    import os
    if not os.path.exists(pdf_path):
        logger.warning(f"[pdf] 파일 없음: {pdf_path}")
        return None

    return RawDocument(
        source_type="분쟁사례",
        source_name=source_name,
        url=f"file://{pdf_path}",     # 로컬 파일 경로를 URI 형식으로 표현
        raw_html="",                   # PDF는 HTML 없음
        extra={"pdf_path": pdf_path},  # 실제 경로는 extra에 보존
    )


# =============================================================================
# 전체 수집 진입점
# =============================================================================

def load_local_raw() -> list[RawDocument]:
    """data/raw에 저장된 데모 원본 파일을 RawDocument로 로드한다."""
    docs: list[RawDocument] = []
    if not RAW_DIR.exists():
        return docs

    for path in sorted(RAW_DIR.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"[raw] UTF-8 로드 실패, 스킵: {path}")
            continue

        name = unicodedata.normalize("NFC", path.name)
        if name.startswith("사규_"):
            docs.append(RawDocument(
                source_type="사규",
                source_name=path.stem.removeprefix("사규_"),
                url=f"file://{path}",
                raw_html=content,
                extra={"source_path": str(path), "format": "kofia_html"},
            ))
        elif "법규" in name:
            docs.append(RawDocument(
                source_type="법규",
                source_name="자본시장과금융투자업에관한법률",
                url=f"file://{path}",
                raw_html=content,
                extra={"source_path": str(path), "format": "drf_xml"},
            ))

    prec_dir = RAW_DIR / "prec"
    if prec_dir.exists():
        for path in sorted(prec_dir.glob("*.xml")):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"[raw] UTF-8 로드 실패, 스킵: {path}")
                continue
            docs.append(RawDocument(
                source_type="분쟁사례",
                source_name="법원판례",
                url=f"file://{path}",
                raw_html=content,
                extra={"source_path": str(path), "format": "prec_xml", "prec_id": path.stem},
            ))

    logger.info(f"[raw] 로컬 RawDocument {len(docs)}개 로드")
    return docs

def scrape_all(pdf_paths: list[str] = None) -> list[RawDocument]:
    """
    모든 소스에서 수집을 실행하고 결과를 합쳐 반환한다.
    일부 소스 실패 시 나머지 결과만으로 계속 진행한다.

    Args:
        pdf_paths: PDF 파일 경로 리스트 (없으면 None)

    Returns:
        수집된 RawDocument 리스트
    """
    docs = load_local_raw()
    if not docs:
        docs.extend(scrape_kofia())    # 1. 사규
        docs.extend(scrape_lawgokr())  # 2. 법규
        docs.extend(scrape_prec())     # 3. 판례

    # 4. PDF (사용자가 파일을 제공한 경우에만)
    if pdf_paths:
        for path in pdf_paths:
            doc = load_pdf_as_raw(path)
            if doc:
                docs.append(doc)

    logger.info(f"총 {len(docs)}개 RawDocument 수집 완료")
    return docs
