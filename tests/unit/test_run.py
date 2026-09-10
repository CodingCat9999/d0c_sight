"""파이프라인 실행과 CLI 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from d0c_sight.cli import main
from d0c_sight.domain.models import Block, BlockKind, ChunkLimits, Document, DocVersion, Section
from d0c_sight.pipeline.chunk import DEFAULT_LIMITS, chunk_document
from d0c_sight.pipeline.run import chunk_documents, write_jsonl

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


# ─── JSONL ────────────────────────────────────────────────────


def test_jsonl_round_trips_the_reference(tmp_path: Path) -> None:
    """근거 참조가 온전히 살아남아야 한다. Phase 6 과 프로젝트 2 가 이것을 읽는다."""
    doc = _doc(Block(BlockKind.PROSE, "hello world"))
    out = tmp_path / "c.jsonl"

    write_jsonl(chunk_document(doc), out)

    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    ref = rows[0]["ref"]
    assert ref["source_id"] == "test"
    assert ref["section_id"] == "SEC"
    assert ref["version"] == {"major": "18", "full": "18.6"}
    assert ref["url"].endswith("#SEC")


def test_jsonl_records_the_limits_used(tmp_path: Path) -> None:
    out = tmp_path / "c.jsonl"
    write_jsonl(chunk_document(_doc(Block(BlockKind.PROSE, "x")), SMALL), out)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["limits"]["table_max_bytes"] == SMALL.table_max_bytes


def test_jsonl_writes_one_line_per_chunk(tmp_path: Path) -> None:
    doc = _doc(
        Block(BlockKind.PROSE, "a"),
        Block(BlockKind.CODE, "SELECT 1;", subtype="programlisting"),
    )
    out = tmp_path / "c.jsonl"
    report = write_jsonl(chunk_document(doc), out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == report.chunks


# ─── 보고 ─────────────────────────────────────────────────────


def test_report_counts_documents_and_sections(tmp_path: Path) -> None:
    from d0c_sight.sources.postgres_docs.parse import parse_all

    docs = list(parse_all(FIXTURES, VERSION))
    report = write_jsonl(chunk_documents(docs), tmp_path / "c.jsonl")

    assert report.documents == len(docs)
    assert report.sections == sum(len(d.sections) for d in docs)
    assert report.chunks > report.sections


def test_report_surfaces_split_chunks(tmp_path: Path) -> None:
    """측정 한계를 결과에 담는다(BRIEF §6.6). 조각난 청크가 있으면 그 사실이 보여야 한다."""
    lines = ["| a | b |", "| --- | --- |"] + [f"| r{i} | v{i} |" for i in range(60)]
    doc = _doc(Block(BlockKind.TABLE, "\n".join(lines), heading="T"))

    report = write_jsonl(chunk_document(doc, SMALL), tmp_path / "c.jsonl")

    assert report.split_chunks > 0
    assert any("분할된 청크" in ln for ln in report.as_lines())


def test_report_mentions_oversized_only_when_present(tmp_path: Path) -> None:
    """없는 경고를 내지 않는다. 항상 나오는 경고는 곧 읽히지 않는다."""
    small = write_jsonl(chunk_document(_doc(Block(BlockKind.PROSE, "짧다"))), tmp_path / "a.jsonl")
    assert not any("임계값 초과" in ln for ln in small.as_lines())

    big = _doc(Block(BlockKind.CODE, "x" * 5000, subtype="programlisting"))
    large = write_jsonl(chunk_document(big), tmp_path / "b.jsonl")
    assert any("임계값 초과" in ln for ln in large.as_lines())


def test_report_always_states_the_limits(tmp_path: Path) -> None:
    """'무엇을 재고 있는가'는 조건 없이 항상 낸다(BRIEF §6.5 예외 조항)."""
    report = write_jsonl(chunk_document(_doc(Block(BlockKind.PROSE, "x"))), tmp_path / "c.jsonl")
    assert any(str(DEFAULT_LIMITS.max_chars) in ln for ln in report.as_lines())


# ─── CLI ──────────────────────────────────────────────────────


def test_cli_ingest_succeeds_on_fixtures(tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    code = main(["ingest", "--html-dir", str(FIXTURES), "--out", str(out)])

    assert code == 0
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) > 100


def test_cli_fails_on_missing_html_dir(tmp_path: Path) -> None:
    code = main(
        ["ingest", "--html-dir", str(tmp_path / "nope"), "--out", str(tmp_path / "o.jsonl")]
    )
    assert code == 1


def test_cli_fails_when_no_chunks_were_produced(tmp_path: Path) -> None:
    """빈 결과를 성공으로 보고하지 않는다. 0개 통과는 통과가 아니다."""
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["ingest", "--html-dir", str(empty), "--out", str(tmp_path / "o.jsonl")])
    assert code == 1
