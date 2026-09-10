"""PostgreSQL DocBook HTML 파싱.

표준 라이브러리 `xml.etree.ElementTree` 로 파싱한다. lxml 이 필요하지 않은 이유는
실측했기 때문이다 — 문서 1,148개 전부가 well-formed XHTML 1.0 Transitional 로
ElementTree 파싱에 성공한다. 의존성 하나를 아끼는 것보다, 관대한 HTML 파서가 조용히
복구해버리는 구조 오류를 예외로 드러내는 쪽이 낫다.

구조를 텍스트로 되돌린 뒤 추측하지 않는다. 표는 행 경계가 보존되도록 직렬화하고,
코드 블록은 공백을 보존하며, 정의 목록의 항목은 각각 독립된 블록이 된다(ADR 0003).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import TYPE_CHECKING

from d0c_sight.domain.models import Block, BlockKind, Document, Section

if TYPE_CHECKING:
    from pathlib import Path

    from d0c_sight.domain.models import DocVersion

SOURCE_ID = "postgres_docs"

NS = "{http://www.w3.org/1999/xhtml}"

#: 섹션 컨테이너로 취급하는 div class. 실측 결과 이들 4,023개 전부가 id 를 가진다.
SECTION_CLASSES = frozenset({
    "book", "part", "preface", "chapter", "appendix", "article",
    "sect1", "sect2", "sect3", "sect4", "sect5",
    "refentry", "refsect1", "refsect2", "refsect3",
    "index", "glossary", "bibliography",
})  # fmt: skip

#: 콘텐츠가 아닌 div. 목차와 네비게이션이 인덱스에 들어가면 검색 품질을 해친다.
SKIPPED_CLASSES = frozenset({"titlepage", "toc", "navheader", "navfooter", "toc-title"})

#: 인덱싱에서 제외하는 문서. 색인·목차 성격이라 섹션 구조가 아예 없다(ADR 0003).
EXCLUDED_DOCS = frozenset({
    "bookindex", "index", "sql-commands",
    "reference-client", "reference-server",
    "biblio", "legalnotice",
})  # fmt: skip

_PROSE_TAGS = frozenset({f"{NS}p", f"{NS}ul", f"{NS}ol", f"{NS}blockquote"})
_CELL_TAGS = frozenset({f"{NS}td", f"{NS}th"})
_HEADINGS = tuple(f"{NS}h{n}" for n in range(1, 7))


def doc_url(doc_id: str, version: DocVersion, section_id: str | None = None) -> str:
    """공식 문서 URL. 근거 링크가 이 형태로 사용자에게 나간다."""
    base = f"https://www.postgresql.org/docs/{version.major}/{doc_id}.html"
    return f"{base}#{section_id}" if section_id else base


#: DocBook 이 제목마다 붙이는 자기 참조 앵커(`<a class="id_link">#</a>`).
#: 텍스트에 섞이면 모든 제목이 " #" 로 끝난다.
_SKIP_INLINE_CLASSES = frozenset({"id_link"})


def _itertext(el: ET.Element) -> Iterator[str]:
    """itertext() 와 같되 네비게이션용 인라인 요소를 건너뛴다."""
    if el.text:
        yield el.text
    for child in el:
        if child.get("class") not in _SKIP_INLINE_CLASSES:
            yield from _itertext(child)
        if child.tail:
            yield child.tail


def _flat(el: ET.Element) -> str:
    """공백을 정규화한 텍스트. 산문과 표 셀에 쓴다."""
    return " ".join("".join(_itertext(el)).split())


def _verbatim(el: ET.Element) -> str:
    """공백을 보존한 텍스트. 코드 블록에 쓴다 — 들여쓰기가 의미를 가진다."""
    return "".join(_itertext(el)).strip("\n")


def _title_of(section: ET.Element) -> str:
    for child in section:
        if child.get("class") == "titlepage":
            for h in child.iter():
                if h.tag in _HEADINGS:
                    return _flat(h)
    return ""


def _table_markdown(table: ET.Element) -> str:
    """표를 행 경계가 보존되는 형태로 직렬화한다.

    청킹에서 헤더를 반복하며 행 그룹으로 나눌 수 있어야 하므로, 텍스트로 뭉개지
    않고 행 단위 구조를 남긴다(ADR 0003).
    """
    rows: list[list[str]] = []
    for tr in table.iter(f"{NS}tr"):
        cells = [_flat(c).replace("|", "\\|") for c in tr if c.tag in _CELL_TAGS]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    lines = [f"| {' | '.join(r + [''] * (width - len(r)))} |" for r in rows]
    lines.insert(1, f"| {' | '.join(['---'] * width)} |")
    return "\n".join(lines)


def _table_block(table: ET.Element, heading: str | None) -> Block | None:
    text = _table_markdown(table)
    if not text:
        return None
    return Block(kind=BlockKind.TABLE, text=text, heading=heading or table.get("summary"))


def _definition_blocks(dl: ET.Element) -> Iterator[Block]:
    """`<dl class="variablelist">` 의 dt+dd 쌍을 각각 독립 블록으로 낸다.

    연속된 dt 는 하나의 dd 를 공유하는 동의어이므로 묶는다. 설정 파라미터와 함수
    설명이 모두 이 구조이며, 문서 전체에서 표보다 압도적으로 지배적이다.
    """
    terms: list[str] = []
    for child in dl:
        if child.tag == f"{NS}dt":
            terms.append(_flat(child))
        elif child.tag == f"{NS}dd":
            body = _flat(child)
            if terms and body:
                yield Block(
                    kind=BlockKind.DEFINITION,
                    text=body,
                    heading=" / ".join(t for t in terms if t),
                )
            terms = []


def _iter_blocks(parent: ET.Element) -> Iterator[Block]:
    """섹션의 직속 콘텐츠. 중첩 섹션은 별도 Section 이 되므로 건너뛴다."""
    for child in parent:
        cls = child.get("class") or ""
        tag = child.tag

        if tag == f"{NS}div":
            if cls in SECTION_CLASSES or cls in SKIPPED_CLASSES:
                continue
            if cls in {"table", "informaltable"}:
                heading = next(
                    (_flat(p) for p in child if p.get("class") == "title"),
                    None,
                )
                for tbl in child.iter(f"{NS}table"):
                    block = _table_block(tbl, heading)
                    if block:
                        yield block
                continue
            yield from _iter_blocks(child)

        elif tag == f"{NS}dl":
            if cls == "variablelist":
                yield from _definition_blocks(child)

        elif tag == f"{NS}pre":
            text = _verbatim(child)
            if text:
                yield Block(kind=BlockKind.CODE, text=text, subtype=cls or None)

        elif tag == f"{NS}table":
            block = _table_block(child, None)
            if block:
                yield block

        elif tag in _PROSE_TAGS:
            text = _flat(child)
            if text:
                yield Block(kind=BlockKind.PROSE, text=text)


def _iter_sections(parent: ET.Element, level: int = 1) -> Iterator[Section]:
    for child in parent:
        if child.tag != f"{NS}div":
            continue
        cls = child.get("class") or ""
        if cls not in SECTION_CLASSES:
            if cls not in SKIPPED_CLASSES:
                yield from _iter_sections(child, level)
            continue
        section_id = child.get("id")
        if section_id is None:
            # 실측상 0건이지만, 조용히 근거 없는 청크를 만드느니 건너뛴다.
            continue
        blocks = tuple(_iter_blocks(child))
        if blocks:
            yield Section(
                section_id=section_id,
                title=_title_of(child),
                level=level,
                blocks=blocks,
            )
        yield from _iter_sections(child, level + 1)


def parse_document(path: Path, version: DocVersion) -> Document:
    """HTML 파일 하나를 Document 로 변환한다."""
    root = ET.parse(path).getroot()
    body = root.find(f"{NS}body")
    sections = tuple(_iter_sections(body)) if body is not None else ()
    doc_id = path.stem
    head_title = root.find(f"{NS}head/{NS}title")
    title = _flat(head_title) if head_title is not None else doc_id
    if not title:
        title = doc_id
    return Document(
        source_id=SOURCE_ID,
        doc_id=doc_id,
        title=title,
        version=version,
        url=doc_url(doc_id, version),
        sections=sections,
    )


def parse_all(html_dir: Path, version: DocVersion) -> Iterator[Document]:
    """디렉터리의 모든 문서를 변환한다. 제외 대상은 여기서 걸러진다.

    필터를 상위 계층에 맡기지 않는 이유는, 각자 거르면 한 군데를 빠뜨렸을 때
    조용히 틀리기 때문이다(BRIEF §6.2).
    """
    for path in sorted(html_dir.glob("*.html")):
        if path.stem in EXCLUDED_DOCS:
            continue
        doc = parse_document(path, version)
        if doc.sections:
            yield doc
