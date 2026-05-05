# =============================================================================
# parser.py
# -----------------------------------------------------------------------------
# 역할: RawDocument(원시 HTML/XML/PDF)를 받아 citation 단위 ParsedChunk로 변환한다.
#       이 파일은 데이터 파이프라인의 두 번째 단계이다.
#
# 데이터 흐름:
#   scraper.py → [parser.py] → ingest.py → Vector DB
#
# 핵심 설계 결정:
#   - source별 실제 구조를 사용한다.
#     → DRF XML은 <조문단위>, 판례 XML은 <PrecService>를 파싱
#     → KOFIA HTML은 실제 제목 DOM을 기준으로 본문을 모음
#     → 문서 본문 전체에 regex를 걸어 inline 법조문 참조를 청크로 만들지 않음
#
#   - 스키마 필드를 이 단계에서 모두 채움
#     → ingest.py는 ParsedChunk를 그냥 TextNode로 변환만 하면 됨
#
# ParsedChunk 스키마:
#   doc_id      : 고유 ID (source_type + source_name + citation_id)
#   source_type : "사규" | "법규" | "분쟁사례"  ← 에이전트 필터링 키
#   source_name : 규정명
#   citation_id : exact-match 검증용 표준 인용 ID
#   article_no  : 조항번호 (조항형 문서 전용)
#   section_no  : 섹션번호 (섹션형 문서 전용)
#   case_no     : 사건번호 (분쟁사례 전용)
#   url         : 원본 URL
#   category    : 조항 내용 분류 (적합성원칙, 설명의무 등)
#   keywords    : 검색 보조 키워드 리스트
#   text        : 임베딩할 본문 (벡터 검색의 실질 대상)
# =============================================================================

import re        # 조항 경계 ("제N조") 패턴 매칭에 사용
import logging
import unicodedata
import xml.etree.ElementTree as ET   # 법제처 DRF XML 파싱
from dataclasses import dataclass    # ParsedChunk 정의
from typing import Optional

from bs4 import BeautifulSoup        # HTML에서 텍스트 추출
from scraper import RawDocument      # 파이프라인 앞 단계 결과물

logger = logging.getLogger(__name__)


# =============================================================================
# ParsedChunk: 벡터 DB에 저장될 최소 단위
# "citation 1개 = 청크 1개" 원칙
# =============================================================================
@dataclass
class ParsedChunk:
    """Vector DB에 적재될 최소 단위. citation 1개 = 청크 1개."""

    # 고유 식별자: 중복 적재 방지 및 lookup에 사용
    doc_id:      str

    # "사규" | "법규" | "분쟁사례"
    # tools.py에서 MetadataFilter로 이 값을 필터링한다
    # 예: source_type="사규" → regulation_search만 이 청크를 가져옴
    source_type: str

    # 규정 공식명 (예: "표준투자권유준칙")
    # 최종 답변에서 인용 표시로 사용 (예: "표준투자권유준칙 제5조")
    source_name: str

    # exact-match 검증과 metadata filtering에 쓰는 표준 인용 ID
    citation_id: Optional[str]

    # 조항형 문서 전용 필드
    article_no:  Optional[str]
    article_title: Optional[str]

    # 섹션형 문서 전용 필드
    section_no: Optional[str]
    section_title: Optional[str]

    # 분쟁사례 사건번호 (예: "2022-증권-031")
    # 분쟁사례 전용. 사규/법규는 None
    case_no: Optional[str]

    # 원본 URL — 출처 추적용
    url:         str

    # 조항 내용 분류 (예: "적합성원칙", "설명의무")
    # CATEGORY_MAP 규칙으로 자동 분류됨
    category:    str

    # 검색 보조 키워드 — 벡터 유사도 검색 보완용
    keywords:    list[str]

    # 벡터 임베딩의 실제 대상 텍스트
    # "인용 제목\n본문" 형태로 구성
    text:        str

    verified:     bool = False


def _make_doc_id(source_type: str, source_name: str, identifier: str) -> str:
    """
    고유 doc_id 생성 함수.
    공백과 특수문자를 언더스코어로 대체해서 안전한 ID를 만든다.

    예: "사규_표준투자권유준칙_제5조" → "사규_표준투자권유준칙_제5조" (한글은 유지)
    """
    base = f"{source_type}_{source_name}_{identifier}"
    # \w: 영문/숫자/언더스코어, 가-힣: 한글 — 이 외의 문자는 _로 치환
    return re.sub(r"[^\w가-힣]", "_", base)


# =============================================================================
# 카테고리 분류 규칙
# -----------------------------------------------------------------------------
# 튜플 키: 해당 카테고리를 나타내는 키워드들 (하나라도 본문에 있으면 해당 카테고리)
# 값: 카테고리 레이블
# 순서 중요: 위에서부터 매칭되므로 더 구체적인 규칙을 위에 배치
# =============================================================================
CATEGORY_MAP = {
    ("적합", "권유", "투자자유형"):     "적합성원칙",
    ("설명", "고지", "중요사항"):       "설명의무",
    ("내부통제", "준법", "감시"):       "내부통제",
    ("정보차단", "차이니즈월"):         "정보차단벽",
    ("불건전", "금지행위"):             "불건전영업행위",
    ("고령", "65세", "노령"):           "고령투자자보호",
}


def _classify_category(text: str) -> str:
    """
    텍스트에서 카테고리를 분류한다.
    CATEGORY_MAP의 키워드가 하나라도 포함되면 해당 카테고리를 반환.
    어디에도 해당하지 않으면 "기타" 반환.
    """
    for keywords, category in CATEGORY_MAP.items():
        if any(kw in text for kw in keywords):
            return category
    return "기타"


# 검색에 도움이 되는 금융 도메인 핵심 키워드 목록
# 본문에 포함된 것만 키워드로 추출 (없는 키워드를 만들어내지 않음)
SEARCH_KEYWORDS = [
    "적합성", "설명의무", "투자권유", "일반투자자", "전문투자자",
    "고령투자자", "파생상품", "ELS", "펀드", "내부통제",
    "정보차단벽", "준법감시", "불건전영업", "손실", "원금",
]


def _extract_keywords(text: str) -> list[str]:
    """본문에 실제로 등장하는 키워드만 추출한다."""
    return [kw for kw in SEARCH_KEYWORDS if kw in text]


def _clean_text(text: str) -> str:
    """검색 품질을 위해 과도한 공백과 빈 줄을 줄인다."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _make_parsed_chunk(
    *,
    source_type: str,
    source_name: str,
    identifier: str,
    citation_id: Optional[str],
    article_no: Optional[str],
    case_no: Optional[str],
    url: str,
    text: str,
    article_title: Optional[str] = None,
    section_no: Optional[str] = None,
    section_title: Optional[str] = None,
) -> ParsedChunk:
    """원문 단위 1개를 citation-preserving chunk 1개로 변환한다."""
    doc_id = _make_doc_id(source_type, source_name, identifier)
    cleaned = _clean_text(text)
    return ParsedChunk(
        doc_id=doc_id,
        source_type=source_type,
        source_name=source_name,
        citation_id=citation_id,
        article_no=article_no,
        article_title=article_title,
        section_no=section_no,
        section_title=section_title,
        case_no=case_no,
        url=url,
        category=_classify_category(cleaned),
        keywords=_extract_keywords(cleaned),
        text=cleaned,
        verified=False,
    )


# 조항형 제목 패턴: "제5조", "제5조의2", "제5조(목적)" 등
# ^ $ 앵커로 본문 내 인라인 법조문 참조("...자본시장법 제47조에 따라...")를 걸러낸다
ARTICLE_HEADING_PATTERN = re.compile(
    r"^(제\s*\d+\s*조(?:\s*의\s*\d+)?)(?:\s*\(([^)]*)\))?$"
)

# 섹션형 제목 패턴: "1.1 제정 목적", "2.2.1 준법감시인의 선임 및 해임" 등
# {2,40} 상한: "1.5배를 초과하는 경우에는..." 같은 본문 문장을 제목으로 오인하지 않도록 차단
SECTION_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)\s+(.{2,40})$")

def _normalize_article_heading(heading: str) -> tuple[Optional[str], Optional[str], str]:
    """
    제목 문자열에서 조항번호·제목·표시명을 추출한다.
    반환: (article_no, article_title, display)
      예: "제 5 조(목적)" → ("제5조", "목적", "제5조(목적)")
    """
    cleaned = re.sub(r"\s+", " ", heading).strip()
    match = ARTICLE_HEADING_PATTERN.match(cleaned)
    if not match:
        return None, None, cleaned
    # 조항번호 내부 공백 제거: "제 5 조" → "제5조"
    article_no = re.sub(r"\s+", "", match.group(1))
    article_title = (match.group(2) or "").strip() or None
    display = f"{article_no}({article_title})" if article_title else article_no
    return article_no, article_title, display


def _sections_from_jo_divs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    KOFIA HTML의 div.JO 구조에서 (제목, 본문) 쌍을 추출한다.

    두 KOFIA 문서 모두 동일한 JO div 구조를 사용한다:
      <div class="JO">
        <div class="article">          ← 제목 영역
          <table><tr>
            <td>&nbsp;제N조(제목)</td>  ← 조항/섹션 제목
            <td>[인쇄 버튼]</td>        ← UI 요소 (텍스트 없음, 무시됨)
          </tr></table>
        </div>
        <div class="none">…</div>     ← 산문형 본문 (선택적)
        <div class="hang">…</div>     ← ① ② 항 (선택적)
        <div class="ho">…</div>       ← 1. 2. 호 (선택적)
        <div class="dann">…</div>     ← 가. 나. 목 (선택적)
        <div class="mok">…</div>      ← (1) (2) 세목 (선택적)
      </div>

    td 순회 방식 대신 div.JO를 직접 선택하므로:
      - "1.5배를 초과…" 같은 소수점 숫자 문장을 섹션 제목으로 오인하지 않는다
      - article 구분 경계가 DOM 구조로 보장되어 regex 오탐 불가
      - div.none(산문)과 div.ho/hang(목록) 두 가지 본문 형식을 모두 수집한다
    """
    sections: list[tuple[str, str]] = []

    for jo in soup.find_all("div", class_="JO"):
        # ── 제목 추출: div.article 안의 첫 번째 td 텍스트 ──────────────────
        article_div = jo.find("div", class_="article")
        if not article_div:
            continue

        heading = unicodedata.normalize(
            "NFC",
            article_div.get_text(" ", strip=True).replace("\xa0", " "),
        ).strip()
        if not heading:
            continue

        # ── 본문 수집: div.article을 제외한 모든 자식 div ────────────────────
        # div.none: 산문 설명 / div.hang,ho,dann,mok: 항·호·목 목록 항목
        body_parts: list[str] = []
        for child in jo.children:
            if not child.name:
                continue
            if "article" in (child.get("class") or []):
                continue  # 제목 div는 본문에서 제외
            text = unicodedata.normalize(
                "NFC",
                child.get_text(" ", strip=True).replace("\xa0", " "),
            ).strip()
            if text:
                body_parts.append(text)

        body = "\n".join(body_parts)
        # 본문 20자 미만 → 실질 내용 없는 상위 섹션 (예: "2.2 준법감시인")
        if len(body) >= 20:
            sections.append((heading, body))

    return sections


# =============================================================================
# 사규 파서: kofia HTML → ParsedChunk 리스트
# =============================================================================

def parse_kofia(doc: RawDocument) -> list[ParsedChunk]:
    """
    KOFIA HTML을 실제 DOM 제목 기준으로 파싱한다.
    표준투자권유준칙은 제N조 제목형, 표준내부통제기준은 1.1/2.2.1 섹션형이다.
    """
    soup = BeautifulSoup(doc.raw_html, "html.parser")

    if doc.source_name == "표준투자권유준칙":
        chunks = _parse_kofia_article_html(doc, soup)
    elif doc.source_name == "금융투자회사표준내부통제기준":
        chunks = _parse_kofia_section_html(doc, soup)
    else:
        logger.warning(f"[kofia] 알 수 없는 HTML 구조: {doc.source_name}")
        chunks = []

    logger.info(f"[kofia] {doc.source_name}: {len(chunks)}개 청크 생성")
    return chunks


def _parse_kofia_article_html(doc: RawDocument, soup: BeautifulSoup) -> list[ParsedChunk]:
    """
    표준투자권유준칙 HTML 파싱 (조항형: 제N조).

    _sections_from_jo_divs로 div.JO 단위로 (제목, 본문) 쌍을 추출한 뒤,
    ARTICLE_HEADING_PATTERN으로 제목을 조항번호·제목으로 분리한다.
    """
    chunks = []
    for heading, body in _sections_from_jo_divs(soup):
        heading_clean = re.sub(r"\s+", " ", heading).strip()
        article_no, article_title, display = _normalize_article_heading(heading_clean)
        if not article_no:
            continue  # 조항 패턴 불일치 → 섹션형 문서의 제목 등 → 스킵

        text = f"{display}\n{body}"
        chunks.append(_make_parsed_chunk(
            source_type=doc.source_type,
            source_name=doc.source_name,
            identifier=article_no,   # doc_id·citation_id 키: "제5조"
            citation_id=article_no,
            article_no=article_no,
            article_title=article_title,
            case_no=None,
            url=doc.url,
            text=text,
        ))

    return chunks


def _parse_kofia_section_html(doc: RawDocument, soup: BeautifulSoup) -> list[ParsedChunk]:
    """
    금융투자회사표준내부통제기준 HTML 파싱 (섹션형: 1.1, 2.2.1).

    _sections_from_jo_divs로 div.JO 단위로 (제목, 본문) 쌍을 추출한 뒤,
    SECTION_HEADING_PATTERN으로 섹션번호·섹션제목으로 분리한다.
    {2,40} 상한이 있어 "1.5배를 초과…" 같은 본문 문장이 섞여도 오탐 없다.
    """
    chunks = []
    for heading, body in _sections_from_jo_divs(soup):
        heading_clean = re.sub(r"\s+", " ", heading).strip()
        match = SECTION_HEADING_PATTERN.match(heading_clean)
        if not match:
            continue  # 섹션 패턴 불일치 → 조항형 제목 등 → 스킵

        section_no, section_title = match.groups()
        text = f"{section_no} {section_title}\n{body}"
        chunks.append(_make_parsed_chunk(
            source_type=doc.source_type,
            source_name=doc.source_name,
            identifier=section_no,   # doc_id·citation_id 키: "2.2.1"
            citation_id=section_no,
            article_no=None,
            section_no=section_no,
            section_title=section_title,
            case_no=None,
            url=doc.url,
            text=text,
        ))

    return chunks


# =============================================================================
# 법규 파서: 법제처 DRF XML → ParsedChunk 리스트
# =============================================================================

def parse_lawgokr(doc: RawDocument) -> list[ParsedChunk]:
    """
    법제처 DRF XML을 파싱해 조항 단위 청크를 생성한다.
    extra["format"]="drf_xml"인 경우 XML 파싱, 아닌 경우 HTML fallback.
    """
    if doc.extra.get("format") == "drf_xml":
        return _parse_drf_xml(doc)

    logger.warning(f"[law.go.kr] 구조화되지 않은 법규 문서 스킵: {doc.source_name}")
    return []


def _parse_drf_xml(doc: RawDocument) -> list[ParsedChunk]:
    """
    법제처 DRF XML 구조 파싱.

    XML 구조:
      <법령>
        <조문>
          <조문단위>
            <조문번호>1</조문번호>
            <조문여부>조문</조문여부>   ← "조문"인 것만 실제 조항
            <조문제목>목적</조문제목>
            <조문내용><![CDATA[...]]></조문내용>
            <항>
              <항번호>①</항번호>
              <항내용><![CDATA[...]]></항내용>
              <호>
                <호번호>1.</호번호>
                <호내용><![CDATA[...]]></호내용>
                <목>...</목>
              </호>
            </항>
          </조문단위>
        </조문>
      </법령>

    조문내용과 실제 DRF 태그인 항/호/목 내용을 순서대로 합쳐 본문을 구성한다.
    """
    try:
        root = ET.fromstring(doc.raw_html.encode("utf-8"))
    except ET.ParseError as e:
        logger.warning(f"[DRF XML] 파싱 실패 ({doc.source_name}): {e}")
        return []

    chunks = []
    for unit in root.findall(".//조문단위"):
        # "전문"(법 전문/편장 제목) 등 실제 조항이 아닌 항목 제외
        yeobu_el = unit.find("조문여부")
        if yeobu_el is None or yeobu_el.text != "조문":
            continue

        no_el = unit.find("조문번호")
        branch_el = unit.find("조문가지번호")
        title_el = unit.find("조문제목")
        content_el = unit.find("조문내용")

        if no_el is None:
            continue

        no = (no_el.text or "").strip()
        branch_no = (branch_el.text or "").strip() if branch_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""

        article_no = f"제{no}조"
        if branch_no and branch_no != "0":
            article_no += f"의{branch_no}"
        article_label = f"{article_no}({title})" if title else article_no

        def _with_label(label: str, body_text: str) -> str:
            body_text = _clean_text(body_text)
            if not label:
                return body_text
            return body_text if body_text.startswith(label) else f"{label} {body_text}".strip()

        def _append_child_text(parent: ET.Element, child_tag: str, no_tag: str, content_tag: str, out: list[str]) -> None:
            for child in parent.findall(child_tag):
                label = (child.findtext(no_tag) or "").strip()
                content = (child.findtext(content_tag) or "").strip()
                if content:
                    out.append(_with_label(label, content))
                if child_tag in {"항", "항단위"}:
                    _append_child_text(child, "호", "호번호", "호내용", out)
                if child_tag == "호":
                    _append_child_text(child, "목", "목번호", "목내용", out)

        # 본문: 조문내용 + 항/호/목 내용 순서대로 합침
        # 법제처 DRF XML은 버전에 따라 <항> 또는 <항단위> 태그를 사용한다.
        # 둘 다 무조건 호출하면 미래 버전 XML이 두 태그를 모두 포함할 때 본문이 중복된다.
        # → 실제로 존재하는 태그 하나만 선택해서 호출한다.
        parts = []
        if content_el is not None and content_el.text:
            parts.append(content_el.text.strip())
        hang_tag = "항" if unit.find("항") is not None else "항단위"
        _append_child_text(unit, hang_tag, "항번호", "항내용", parts)

        body = "\n".join(parts)
        if len(body) < 20:
            continue

        body_clean = _clean_text(body)
        text = body_clean if body_clean.startswith(article_label) else f"{article_label}\n{body_clean}"
        chunks.append(_make_parsed_chunk(
            source_type=doc.source_type,
            source_name=doc.source_name,
            identifier=article_no,
            citation_id=article_no,
            article_no=article_no,
            article_title=title or None,
            case_no=None,
            url=doc.url,
            text=text,
        ))

    logger.info(f"[law.go.kr DRF] {doc.source_name}: {len(chunks)}개 청크 생성")
    return chunks


# =============================================================================
# 분쟁사례 파서: 판례 XML (DRF) 또는 PDF → ParsedChunk 리스트
# =============================================================================

def parse_사례(doc: RawDocument) -> list[ParsedChunk]:
    """분쟁사례 파서 진입점. extra["format"]으로 하위 파서를 선택한다."""
    if doc.extra.get("format") == "prec_xml":
        return _parse_prec_xml(doc)
    return _parse_pdf(doc)


def _parse_prec_xml(doc: RawDocument) -> list[ParsedChunk]:
    """
    법제처 DRF 판례 XML (target=prec) 파싱.

    XML 구조:
      <PrecService>
        <사건번호>2013나2021183-1</사건번호>
        <사건명><![CDATA[손해배상(기)]]></사건명>
        <법원명>서울고등법원</법원명>
        <선고일자>20150423</선고일자>
        <판시사항><![CDATA[...]]></판시사항>  ← 없을 수 있음
        <판결요지><![CDATA[...]]></판결요지>  ← 없을 수 있음
        <판례내용><![CDATA[HTML 전문...]]></판례내용>
      </PrecService>

    판례 1건 = 청크 1개. 판례내용 HTML 태그는 BeautifulSoup으로 제거.
    """
    try:
        root = ET.fromstring(doc.raw_html.encode("utf-8"))
    except ET.ParseError as e:
        logger.warning(f"[prec XML] 파싱 실패: {e}")
        return []

    def _text(tag: str) -> str:
        el = root.find(tag)
        return (el.text or "").strip() if el is not None else ""

    case_no = _text("사건번호")
    case_name = _text("사건명")
    court = _text("법원명")
    date = _text("선고일자")

    # 판시사항·판결요지는 없거나 빈 경우가 있음 — 있으면 포함
    판시 = BeautifulSoup(_text("판시사항"), "html.parser").get_text(" ").strip()
    요지 = BeautifulSoup(_text("판결요지"), "html.parser").get_text(" ").strip()

    # 판례내용은 HTML — 태그 제거 후 본문 추출
    content_raw = _text("판례내용")
    content = BeautifulSoup(content_raw, "html.parser").get_text("\n").strip() if content_raw else ""

    header = f"[{court}] {case_no} ({case_name}, {date[:4]}년)"
    parts = [header]
    if 판시:
        parts.append(f"판시사항: {판시}")
    if 요지:
        parts.append(f"판결요지: {요지}")
    if content:
        parts.append(content)

    text = "\n\n".join(parts)
    if len(text) < 50:
        logger.warning(f"[prec XML] 내용 빈약, 건너뜀: {case_no}")
        return []

    parsed_chunk = _make_parsed_chunk(
        source_type=doc.source_type,
        source_name=doc.source_name,
        identifier=case_no,
        citation_id=case_no,
        article_no=None,
        case_no=case_no,
        url=doc.url,
        text=text,
    )
    logger.info(f"[prec XML] {case_no} ({court}): 청크 생성")
    return [parsed_chunk]


def _parse_pdf(doc: RawDocument) -> list[ParsedChunk]:
    """
    금감원 분쟁사례 PDF를 파싱해 사례 단위 청크를 생성한다.

    PDF 구조 가정 (금감원 분쟁사례 표준 형식):
      각 사례가 "사례 N" 또는 "■" 또는 사건번호로 시작하는 섹션으로 구분됨
      예: "사례 1\n사건번호: 2022-증권-031\n신청인: ...\n..."

    pdfminer.six 라이브러리 사용 (pip install pdfminer.six)
    """
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        logger.error("pdfminer.six 미설치. pip install pdfminer.six 실행 필요")
        return []

    pdf_path = doc.extra.get("pdf_path", "")
    try:
        full_text = extract_text(pdf_path)
    except Exception as e:
        logger.warning(f"[pdf] 파싱 실패 ({pdf_path}): {e}")
        return []

    case_pattern = re.compile(
        r"(사례\s*\d+|■\s*사례|[가-힣\d]+-[가-힣\d]+-\d+)"
    )
    parts = case_pattern.split(full_text)

    chunks = []
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if len(body) < 50:
            i += 2
            continue

        case_no_match = re.search(r"[\d가-힣]+-[가-힣\d]+-\d+", body)
        case_no = case_no_match.group() if case_no_match else header

        text = f"{header}\n{body}"
        chunks.append(_make_parsed_chunk(
            source_type=doc.source_type,
            source_name=doc.source_name,
            identifier=case_no,
            citation_id=case_no,
            article_no=None,
            case_no=case_no,
            url=doc.url,
            text=text,
        ))
        i += 2

    logger.info(f"[pdf] {doc.source_name}: {len(chunks)}개 청크 생성")
    return chunks


# =============================================================================
# 파서 라우터 테이블
# -----------------------------------------------------------------------------
# source_type → 파서 함수 매핑
# 새 데이터 소스 추가 시 이 딕셔너리에만 항목을 추가하면 된다
# =============================================================================
PARSERS = {
    "사규":     parse_kofia,
    "법규":     parse_lawgokr,
    "분쟁사례": parse_사례,
}


def parse_all(raw_docs: list[RawDocument]) -> list[ParsedChunk]:
    """
    RawDocument 리스트 전체를 파싱한다.
    source_type에 맞는 파서를 PARSERS에서 선택해 호출한다.

    Args:
        raw_docs: scraper.scrape_all()의 반환값

    Returns:
        모든 소스에서 생성된 ParsedChunk 리스트
    """
    chunks = []
    for doc in raw_docs:
        # source_type으로 파서 함수 선택
        parser = PARSERS.get(doc.source_type)
        if not parser:
            # 알 수 없는 source_type이면 경고 후 건너뜀
            logger.warning(f"파서 없음: {doc.source_type}")
            continue
        chunks.extend(parser(doc))

    logger.info(f"총 {len(chunks)}개 청크 파싱 완료")
    return chunks
