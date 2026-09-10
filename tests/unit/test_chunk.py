"""청킹 테스트.

경계값을 명시적으로 본다(N-1, N, N+1). 그리고 "자르지 않기로 한 것"이 실제로
잘리지 않는지를 확인한다 — 그 약속이 깨지면 코드 조각과 헤더 없는 표가 인덱스에
들어가고, 검색에서 그것을 알아챌 방법이 없다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from d0c_sight.domain.models import (
    Block,
    BlockKind,
    ChunkLimits,
    Document,
    DocVersion,
    Section,
)
from d0c_sight.pipeline.chunk import DEFAULT_LIMITS, chunk_document
from d0c_sight.sources.postgres_docs.parse import parse_document

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "postgres_docs"
VERSION = DocVersion(major="18", full="18.6")

SMALL = ChunkLimits(target_chars=100, max_chars=100, table_max_bytes=80)


def _doc(*blocks: Block) -> Document:
    return Document(
        source_id="test",
        doc_id="doc",
        title="T",
        version=VERSION,
        url="https://example.test/doc.html",
        sections=(Section(section_id="SEC", title="S", level=1, blocks=blocks),),
    )


def _table(rows: int, cell: str = "x") -> Block:
    lines = ["| a | b |", "| --- | --- |"]
    lines += [f"| {cell}{i} | v{i} |" for i in range(rows)]
    return Block(kind=BlockKind.TABLE, text="\n".join(lines), heading="T1")


# ─── 참조 ─────────────────────────────────────────────────────


def test_every_chunk_carries_the_section_anchor() -> None:
    """근거 추적의 전제. 앵커 없는 청크는 출처를 잃는다."""
    chunks = list(chunk_document(_doc(Block(BlockKind.PROSE, "hello"))))

    assert chunks
    for c in chunks:
        assert c.ref.section_id == "SEC"
        assert c.ref.url.endswith("#SEC")
        assert c.ref.source_id == "test"


def test_limits_are_recorded_on_every_chunk() -> None:
    """임계값을 결과에 실어 보낸다(BRIEF §6.5)."""
    chunks = list(chunk_document(_doc(Block(BlockKind.PROSE, "hi")), SMALL))
    assert all(c.limits == SMALL for c in chunks)


def test_chunk_ids_are_unique_within_a_document() -> None:
    doc = _doc(
        Block(BlockKind.PROSE, "a"),
        Block(BlockKind.CODE, "SELECT 1;", subtype="programlisting"),
        Block(BlockKind.PROSE, "b"),
    )
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids)), ids


# ─── 자르지 않기로 한 것 ──────────────────────────────────────


def test_code_is_never_split_even_when_oversized() -> None:
    """실행할 수 없는 조각을 남기지 않는다(ADR 0003).

    본문에 마침표를 넣은 것은 의도적이다. 마침표가 없으면 산문 분할기가 문장
    경계를 찾지 못해 어차피 안 잘리고, 그러면 이 테스트는 코드 보호가 없어도
    통과한다. 실제로 처음에 그런 테스트였고 변이 검증에서 드러났다.
    """
    body = "\n".join(f"SELECT {i}.0 AS v; -- step {i}. next." for i in range(300))
    chunks = list(
        chunk_document(_doc(Block(BlockKind.CODE, body, subtype="programlisting")), SMALL)
    )

    assert len(chunks) == 1
    assert chunks[0].text == body
    assert chunks[0].is_split is False
    assert len(chunks[0].text) > SMALL.max_chars  # 임계값을 넘겼는데도 그대로다


def test_a_single_long_sentence_is_not_cut_mid_sentence() -> None:
    sentence = "word " * 200
    chunks = list(chunk_document(_doc(Block(BlockKind.PROSE, sentence.strip())), SMALL))
    assert len(chunks) == 1


# ─── 표 분할 ──────────────────────────────────────────────────


def test_small_table_stays_whole() -> None:
    chunks = list(chunk_document(_doc(_table(2))))
    assert len(chunks) == 1
    assert chunks[0].is_split is False


def test_large_table_splits_and_every_part_keeps_the_header() -> None:
    """헤더 없는 행 조각은 검색에도 답변에도 쓸 수 없다."""
    chunks = list(chunk_document(_doc(_table(60)), SMALL))

    assert len(chunks) > 1
    for c in chunks:
        lines = c.text.split("\n")
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"
        assert len(lines) > 2, "헤더만 있고 행이 없는 조각이 나왔다"


def test_split_table_parts_are_numbered_and_complete() -> None:
    chunks = list(chunk_document(_doc(_table(60)), SMALL))
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.of == len(chunks) for c in chunks)
    assert all(c.is_split for c in chunks)


def test_split_table_loses_no_rows() -> None:
    """분할이 데이터를 삼키지 않는지 확인한다."""
    original = _table(60)
    chunks = list(chunk_document(_doc(original), SMALL))

    seen = [ln for c in chunks for ln in c.text.split("\n")[2:]]
    expected = original.text.split("\n")[2:]
    assert seen == expected


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_table_split_boundary(rows: int) -> None:
    """행이 아주 적을 때도 헤더 규칙이 유지되는지 본다."""
    chunks = list(chunk_document(_doc(_table(rows)), SMALL))
    assert all(c.text.startswith("| a | b |") for c in chunks)


# ─── 산문 병합 ────────────────────────────────────────────────


def test_adjacent_prose_is_merged_up_to_target() -> None:
    blocks = [Block(BlockKind.PROSE, "sentence.") for _ in range(5)]
    chunks = list(chunk_document(_doc(*blocks)))
    assert len(chunks) == 1, "인접 산문이 합쳐지지 않았다"


def test_prose_merging_stops_at_a_non_prose_block() -> None:
    """순서를 보존한다. 코드 앞뒤의 산문이 뒤섞이면 문맥이 깨진다."""
    doc = _doc(
        Block(BlockKind.PROSE, "before"),
        Block(BlockKind.CODE, "SELECT 1;", subtype="programlisting"),
        Block(BlockKind.PROSE, "after"),
    )
    kinds = [c.kind for c in chunk_document(doc)]
    assert kinds == [BlockKind.PROSE, BlockKind.CODE, BlockKind.PROSE]


# ─── 실물 ─────────────────────────────────────────────────────


def test_real_definition_document_produces_findable_chunks() -> None:
    """예시 질문("too many clients")의 정답이 온전한 청크로 나오는가."""
    doc = parse_document(FIXTURES / "runtime-config-connection.html", VERSION)
    chunks = list(chunk_document(doc))

    hits = [c for c in chunks if c.heading and c.heading.startswith("max_connections")]
    assert len(hits) == 1, [c.heading for c in chunks[:10]]
    c = hits[0]
    assert c.is_split is False, "정답이 조각나면 검색이 반쪽만 찾는다"
    assert "concurrent connections" in c.text
    assert c.ref.url.startswith("https://www.postgresql.org/docs/18/")


def test_real_largest_table_splits_with_headers_intact() -> None:
    doc = parse_document(FIXTURES / "sql-keywords-appendix.html", VERSION)
    chunks = [c for c in chunk_document(doc) if c.kind is BlockKind.TABLE]

    assert chunks
    split = [c for c in chunks if c.is_split]
    assert split, "코퍼스 최대 표가 분할되지 않았다"
    for c in split:
        assert c.text.split("\n")[0].startswith("| Key Word")


def test_real_corpus_chunks_respect_the_declared_limits() -> None:
    """한계를 넘는 청크는 code 와 table 뿐이어야 한다 — 둘은 자르지 않기로 했다."""
    doc = parse_document(FIXTURES / "multibyte.html", VERSION)
    over = [c for c in chunk_document(doc) if len(c.text) > DEFAULT_LIMITS.max_chars]
    assert all(c.kind in {BlockKind.CODE, BlockKind.TABLE} for c in over), [
        (c.kind, len(c.text)) for c in over
    ]
