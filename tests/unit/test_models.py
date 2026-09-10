"""도메인 모델의 불변식 테스트.

모델에 검증을 넣었으면 그 검증이 실제로 위반을 잡는지 확인해야 한다. 통과하는
경우만 테스트하면 검증이 없는 것과 구별되지 않는다(BRIEF §6.4).
"""

from __future__ import annotations

import pytest

from d0c_sight.domain.models import BlockKind, Chunk, ChunkLimits, DocVersion, SourceRef

LIMITS = ChunkLimits(target_chars=1200, max_chars=2400, table_max_bytes=8192)


def _ref() -> SourceRef:
    return SourceRef(
        source_id="postgres_docs",
        doc_id="runtime-config-connection",
        section_id="GUC-MAX-CONNECTIONS",
        version=DocVersion(major="18", full="18.6"),
        url="https://www.postgresql.org/docs/18/runtime-config-connection.html#GUC-MAX-CONNECTIONS",
    )


def _chunk(part: int = 1, of: int = 1) -> Chunk:
    return Chunk(
        chunk_id="c1",
        text="max_connections (integer)",
        kind=BlockKind.DEFINITION,
        ref=_ref(),
        limits=LIMITS,
        part=part,
        of=of,
    )


# ─── DocVersion ───────────────────────────────────────────────


def test_version_accepts_matching_major_and_full() -> None:
    v = DocVersion(major="18", full="18.6")
    assert v.major == "18"
    assert v.full == "18.6"


@pytest.mark.parametrize("major,full", [("18", "17.6"), ("18", "180.1"), ("18", "18")])
def test_version_rejects_mismatched_full(major: str, full: str) -> None:
    """major 로 시작하지 않는 full 은 거부한다.

    '18' 과 '180.1' 을 함께 넣은 것은 단순 prefix 비교로는 통과하기 때문이다.
    점까지 포함해 비교해야 한다.
    """
    with pytest.raises(ValueError, match="major"):
        DocVersion(major=major, full=full)


# ─── Chunk part/of 경계 ───────────────────────────────────────


def test_chunk_defaults_to_a_single_whole_part() -> None:
    c = _chunk()
    assert (c.part, c.of) == (1, 1)
    assert c.is_split is False


def test_chunk_marks_itself_split_when_of_exceeds_one() -> None:
    assert _chunk(part=1, of=2).is_split is True
    assert _chunk(part=2, of=2).is_split is True


@pytest.mark.parametrize("part,of", [(1, 1), (1, 2), (2, 2), (3, 3)])
def test_chunk_accepts_valid_part_of(part: int, of: int) -> None:
    assert _chunk(part, of).part == part


@pytest.mark.parametrize("part,of", [(0, 1), (2, 1), (1, 0), (-1, 3), (4, 3)])
def test_chunk_rejects_invalid_part_of(part: int, of: int) -> None:
    """경계값을 명시적으로 본다: part=of 는 유효, part=of+1 은 무효."""
    with pytest.raises(ValueError, match="part/of"):
        _chunk(part, of)


# ─── 불변성 ───────────────────────────────────────────────────


def test_chunk_is_frozen() -> None:
    """청크는 만들어진 뒤 바뀌지 않는다. 근거 추적의 전제다."""
    c = _chunk()
    with pytest.raises(AttributeError):
        c.text = "변조"  # type: ignore[misc]


def test_limits_travel_with_the_chunk() -> None:
    """임계값이 결과에 실려 나가는지 확인한다(BRIEF §6.5)."""
    c = _chunk()
    assert c.limits.target_chars == 1200
    assert c.limits.max_chars == 2400
    assert c.limits.table_max_bytes == 8192
