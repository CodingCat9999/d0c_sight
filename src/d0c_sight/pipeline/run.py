"""파이프라인 실행 — 수집에서 청크까지.

여기까지가 결정론적 계층이다. LLM 은 없다. 같은 tarball 을 넣으면 항상 같은 청크가
나와야 Phase 4~5 에서 전략 변경의 효과를 측정할 수 있다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from d0c_sight.pipeline.chunk import DEFAULT_LIMITS, chunk_document

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from d0c_sight.domain.models import Chunk, ChunkLimits, Document


@dataclass(frozen=True, slots=True)
class RunReport:
    """실행 결과와 그 한계.

    수치만 내보내면 사용자는 그것을 완전한 사실로 읽는다(BRIEF §6.6). 무엇을
    제외했고 무엇이 쪼개졌고 어떤 임계값을 넘겼는지를 함께 낸다.
    """

    documents: int
    sections: int
    chunks: int
    split_chunks: int
    oversized_chunks: int
    limits: ChunkLimits

    def as_lines(self) -> list[str]:
        split_pct = self.split_chunks / self.chunks * 100 if self.chunks else 0.0
        lines = [
            f"문서 {self.documents:,}  섹션 {self.sections:,}  청크 {self.chunks:,}",
            f"임계값  target={self.limits.target_chars} "
            f"max={self.limits.max_chars} table={self.limits.table_max_bytes}",
            f"분할된 청크 {self.split_chunks:,} ({split_pct:.1f}%) "
            f"— 이 청크들은 단독으로 완전하지 않다",
        ]
        if self.oversized_chunks:
            lines.append(
                f"임계값 초과 {self.oversized_chunks:,} — 자르지 않기로 한 코드와 표다(ADR 0003)"
            )
        return lines


def read_jsonl(path: Path) -> Iterator[Chunk]:
    """JSONL 을 Chunk 로 되돌린다.

    write_jsonl 과 짝을 이룬다. 왕복이 손실 없이 되어야 Phase 6 의 근거 추적과
    프로젝트 2 의 학습 데이터가 성립한다.
    """
    from d0c_sight.domain.models import (
        BlockKind,
        Chunk,
        ChunkLimits,
        DocVersion,
        SourceRef,
    )

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = row["ref"]
            yield Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                kind=BlockKind(row["kind"]),
                ref=SourceRef(
                    source_id=ref["source_id"],
                    doc_id=ref["doc_id"],
                    section_id=ref["section_id"],
                    version=DocVersion(**ref["version"]),
                    url=ref["url"],
                ),
                limits=ChunkLimits(**row["limits"]),
                heading=row["heading"],
                subtype=row["subtype"],
                part=row["part"],
                of=row["of"],
            )


def chunk_documents(
    docs: Iterable[Document], limits: ChunkLimits = DEFAULT_LIMITS
) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, limits)


def write_jsonl(chunks: Iterable[Chunk], out: Path) -> RunReport:
    """청크를 JSONL 로 쓰고 보고를 돌려준다.

    JSONL 인 이유는 Phase 4 의 임베딩과 프로젝트 2 의 학습 데이터가 모두 줄 단위로
    읽기 때문이다. 근거 참조(ref)를 통째로 실어 보내 나중에 답변과 근거의 쌍을
    복원할 수 있게 한다.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    docs: set[str] = set()
    sections: set[str] = set()
    total = split = oversized = 0
    limits = DEFAULT_LIMITS

    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            docs.add(chunk.ref.doc_id)
            sections.add(f"{chunk.ref.doc_id}#{chunk.ref.section_id}")
            total += 1
            limits = chunk.limits
            if chunk.is_split:
                split += 1
            if len(chunk.text) > chunk.limits.max_chars:
                oversized += 1

    return RunReport(
        documents=len(docs),
        sections=len(sections),
        chunks=total,
        split_chunks=split,
        oversized_chunks=oversized,
        limits=limits,
    )
