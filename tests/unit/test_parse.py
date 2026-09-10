"""파서 테스트 — 실물 픽스처를 쓴다.

합성 HTML 로만 테스트하면 파서가 실제로 마주치는 구조를 놓친다. DocBook 이
제목마다 붙이는 자기 참조 앵커나 목차용 `<dl>` 같은 것은 상상해서 만들 수 없다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from d0c_sight.domain.models import Block, BlockKind, Document, DocVersion
from d0c_sight.sources.postgres_docs.parse import (
    EXCLUDED_DOCS,
    doc_url,
    parse_all,
    parse_document,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "postgres_docs"
VERSION = DocVersion(major="18", full="18.6")


def _doc(name: str) -> Document:
    return parse_document(FIXTURES / f"{name}.html", VERSION)


def _blocks(name: str, kind: BlockKind | None = None) -> list[Block]:
    return [b for s in _doc(name).sections for b in s.blocks if kind is None or b.kind is kind]


def test_fixtures_are_present() -> None:
    """부재 단언 앞의 존재 단언. 픽스처가 없으면 아래가 전부 공허해진다."""
    found = sorted(p.stem for p in FIXTURES.glob("*.html"))
    assert len(found) >= 6, found


# ─── 제목 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("errcodes-appendix", "Appendix A. PostgreSQL Error Codes"),
        ("runtime-config-connection", "19.3. Connections and Authentication"),
    ],
)
def test_document_title_comes_from_head(name: str, expected: str) -> None:
    assert _doc(name).title == expected


def test_titles_do_not_contain_the_self_anchor() -> None:
    """DocBook 은 제목마다 `<a class="id_link">#</a>` 를 붙인다.

    걸러내지 않으면 모든 제목과 정의 항목이 ' #' 로 끝난다.
    """
    doc = _doc("runtime-config-connection")
    assert not doc.title.endswith("#")
    headings = [b.heading for s in doc.sections for b in s.blocks if b.heading]
    assert headings, "정의 항목이 하나도 없다 — 이 단언이 공허하다"
    assert not [h for h in headings if h.endswith("#")]


# ─── 섹션과 앵커 ──────────────────────────────────────────────


def test_every_section_has_an_anchor() -> None:
    """근거 추적이 절 단위로 가능하다는 Phase 6 의 전제다."""
    for name in ("errcodes-appendix", "runtime-config-connection", "multibyte"):
        doc = _doc(name)
        assert doc.sections, f"{name}: 섹션이 없다"
        assert all(s.section_id for s in doc.sections)


def test_doc_url_appends_the_anchor() -> None:
    assert doc_url("errcodes-appendix", VERSION) == (
        "https://www.postgresql.org/docs/18/errcodes-appendix.html"
    )
    assert doc_url("errcodes-appendix", VERSION, "ERRCODES-APPENDIX").endswith("#ERRCODES-APPENDIX")


# ─── 블록 종류별 ──────────────────────────────────────────────


def test_definition_blocks_carry_the_term_as_heading() -> None:
    defs = _blocks("runtime-config-connection", BlockKind.DEFINITION)
    assert len(defs) > 20, f"정의 항목이 {len(defs)}개뿐이다"
    headings = [b.heading for b in defs if b.heading]
    assert any(h.startswith("max_connections") for h in headings), headings[:5]


def test_toc_is_not_mistaken_for_a_definition_list() -> None:
    """`<dl class="toc">` 도 dt/dd 구조다. class 를 보지 않으면 목차가 내용이 된다."""
    defs = _blocks("runtime-config-connection", BlockKind.DEFINITION)
    assert defs, "정의 항목이 없어 이 단언이 공허하다"
    assert not [b for b in defs if b.heading and b.heading.startswith("19.3.1.")]


def test_table_is_serialised_with_row_boundaries() -> None:
    tables = _blocks("errcodes-appendix", BlockKind.TABLE)
    assert len(tables) == 1
    lines = tables[0].text.split("\n")
    assert lines[0].startswith("| Error Code")
    assert set(lines[1].replace("|", "").replace("-", "").strip()) == set()
    assert len(lines) > 300, f"{len(lines)}행 — 표가 잘렸다"
    assert any("23505" in ln for ln in lines)


def test_code_blocks_keep_their_subtype() -> None:
    """programlisting / synopsis / screen 은 서로 다른 질문에 답한다."""
    codes = _blocks("xfunc-sql", BlockKind.CODE)
    assert codes, "코드 블록이 없다"
    assert {c.subtype for c in codes} <= {"programlisting", "synopsis", "screen", "literallayout"}
    assert "programlisting" in {c.subtype for c in codes}


def test_code_blocks_preserve_indentation() -> None:
    """공백을 정규화하면 코드가 망가진다."""
    codes = _blocks("xfunc-sql", BlockKind.CODE)
    assert any("\n" in c.text for c in codes), "여러 줄 코드가 하나도 없다"


def test_mixed_document_yields_both_tables_and_code() -> None:
    kinds = {b.kind for b in _blocks("multibyte")}
    assert BlockKind.TABLE in kinds
    assert BlockKind.CODE in kinds


# ─── 제외 ─────────────────────────────────────────────────────


def test_excluded_documents_are_filtered_by_the_source() -> None:
    """필터를 상위 계층에 맡기지 않는다(BRIEF §6.2)."""
    assert "legalnotice" in EXCLUDED_DOCS
    doc_ids = {d.doc_id for d in parse_all(FIXTURES, VERSION)}
    assert doc_ids, "아무 문서도 나오지 않았다 — 이 단언이 공허하다"
    assert "legalnotice" not in doc_ids
    assert "errcodes-appendix" in doc_ids
