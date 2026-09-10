"""청킹 — 구조가 원자 단위를 정한다(ADR 0003).

LLM 을 쓰지 않는다. 같은 입력에 항상 같은 출력이 나와야 Phase 4~5 에서 청킹 전략
변경의 효과를 측정 비교할 수 있다.

임계값은 실측 분포에서 **일부만 걸리도록** 잡았다. 전부 걸리면 잡음이고 하나도 안
걸리면 검증된 적 없는 코드다(BRIEF §6.5). 각 값의 뜻은 상수 주석에 한 문장으로 적었고,
값 자체는 ChunkLimits 에 실려 결과와 함께 나간다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from d0c_sight.domain.models import BlockKind, Chunk, ChunkLimits

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from d0c_sight.domain.models import Block, Document, Section, SourceRef

#: 산문을 모을 때 목표로 삼는 크기. 정의목록 항목의 p90(762자)을 하나로 담고도 여유가 있다.
TARGET_CHARS = 1200

#: 이보다 큰 원자 단위만 쪼갠다. 실측상 정의 블록의 1.0%(45개)만 해당하므로 분할은 예외다.
MAX_CHARS = 2400

#: 직렬화된 표가 이보다 크면 행 그룹으로 나눈다. 표 486개 중 9.3%(45개)만 대상이 된다.
#: 원본 HTML 바이트가 아니라 마크다운 직렬화 후 크기다 — 태그가 빠지면 크기가 크게 준다.
TABLE_MAX_BYTES = 4096

DEFAULT_LIMITS = ChunkLimits(
    target_chars=TARGET_CHARS,
    max_chars=MAX_CHARS,
    table_max_bytes=TABLE_MAX_BYTES,
)

#: 문장 경계. 산문을 쪼갤 때만 쓴다. 코드와 표는 이 경로로 오지 않는다.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_TABLE_HEADER_LINES = 2  # 헤더 행 + 구분선


def _pack(parts: Sequence[str], limit: int) -> list[str]:
    """조각들을 limit 을 넘지 않게 순서대로 묶는다. 조각 자체는 쪼개지 않는다."""
    out: list[str] = []
    buf = ""
    for part in parts:
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= limit:
            buf = f"{buf} {part}"
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def _split_prose(text: str, limit: int) -> list[str]:
    """산문을 문장 경계로 나눈다. 한 문장이 limit 을 넘으면 그 문장은 그대로 둔다.

    문장 중간을 자르면 의미가 파괴되므로, 크기를 지키는 것보다 문장을 지키는 쪽을
    택한다. 그런 문장이 나왔다는 사실은 part/of 로 드러난다.
    """
    if len(text) <= limit:
        return [text]
    return _pack(_SENTENCE_END.split(text), limit) or [text]


def _split_table(text: str, max_bytes: int) -> list[str]:
    """표를 행 그룹으로 나누되 헤더를 각 조각에 반복한다.

    헤더 없는 행 조각은 만들지 않는다. `| 23505 | unique_violation |` 만 있고
    무엇의 표인지 모르면 검색에도 답변에도 쓸 수 없다.
    """
    if len(text.encode()) <= max_bytes:
        return [text]
    lines = text.split("\n")
    if len(lines) <= _TABLE_HEADER_LINES:
        return [text]
    header = lines[:_TABLE_HEADER_LINES]
    header_bytes = len("\n".join(header).encode()) + 1
    budget = max(max_bytes - header_bytes, 1)

    out: list[str] = []
    group: list[str] = []
    used = 0
    for row in lines[_TABLE_HEADER_LINES:]:
        size = len(row.encode()) + 1
        if group and used + size > budget:
            out.append("\n".join(header + group))
            group, used = [], 0
        group.append(row)
        used += size
    if group:
        out.append("\n".join(header + group))
    return out


def _split_block(block: Block, limits: ChunkLimits) -> list[str]:
    """블록을 청크 본문들로 나눈다. 대부분은 나뉘지 않고 하나로 나온다."""
    if block.kind is BlockKind.TABLE:
        return _split_table(block.text, limits.table_max_bytes)
    if block.kind is BlockKind.CODE:
        # 코드는 절대 중간에서 자르지 않는다(ADR 0003). 실행할 수 없는 조각이 남는다.
        return [block.text]
    return _split_prose(block.text, limits.max_chars)


def _make(
    ref: SourceRef,
    seq: int,
    text: str,
    block: Block,
    limits: ChunkLimits,
    part: int,
    of: int,
) -> Chunk:
    suffix = f".{part}" if of > 1 else ""
    return Chunk(
        chunk_id=f"{ref.doc_id}#{ref.section_id}:{seq}{suffix}",
        text=text,
        kind=block.kind,
        ref=ref,
        limits=limits,
        heading=block.heading,
        subtype=block.subtype,
        part=part,
        of=of,
    )


def chunk_section(
    doc: Document, section: Section, limits: ChunkLimits = DEFAULT_LIMITS
) -> Iterator[Chunk]:
    """한 섹션의 청크들. 모든 청크가 이 섹션의 앵커를 물고 나간다.

    이 함수는 문서가 어느 소스에서 왔는지 모른다. 참조 조립은 도메인 모델이 한다.
    """
    ref = doc.ref_for(section)
    seq = 0
    prose: list[Block] = []

    def flush() -> Iterator[Chunk]:
        nonlocal seq, prose
        if not prose:
            return
        merged = _pack([b.text for b in prose], limits.target_chars)
        head = prose[0]
        prose = []
        for text in merged:
            parts = _split_prose(text, limits.max_chars)
            for i, body in enumerate(parts, start=1):
                yield _make(ref, seq, body, head, limits, i, len(parts))
            seq += 1

    for block in section.blocks:
        if block.kind is BlockKind.PROSE:
            prose.append(block)
            continue
        yield from flush()
        parts = _split_block(block, limits)
        for i, body in enumerate(parts, start=1):
            yield _make(ref, seq, body, block, limits, i, len(parts))
        seq += 1
    yield from flush()


def chunk_document(doc: Document, limits: ChunkLimits = DEFAULT_LIMITS) -> Iterator[Chunk]:
    for section in doc.sections:
        yield from chunk_section(doc, section, limits)
