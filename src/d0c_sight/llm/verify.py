"""근거 검증 — "근거 없는 답변은 내지 않는다"를 코드로 강제한다.

모델이 댄 근거를 그대로 믿지 않는다. 세 가지를 확인한다.

1. chunk_id 가 실재하는가
2. 그 청크가 실제로 컨텍스트에 있었는가 (보지 않은 것을 인용했는가)
3. quote 가 그 청크 본문에 실제로 있는 문자열인가 (변형·창작했는가)

걸러진 근거는 버리지 않고 결과에 실어 보낸다. 조용히 버리면 사용자는 답변이 온전히
검증된 것으로 읽는다(BRIEF §6.6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from d0c_sight.domain.models import (
    Cause,
    Confidence,
    DroppedEvidence,
    DropReason,
    Evidence,
)

if TYPE_CHECKING:
    from collections.abc import Container, Mapping, Sequence

    from d0c_sight.domain.models import Chunk

#: 모델이 id 주위에 붙이는 장식. 표기를 고쳐 원인은 없앴고, 이것은 안전망이다.
_ID_TRIM = "[](){}<>\"'`,. \t\n"

#: 이보다 짧은 인용은 근거로 치지 않는다. "the" 같은 조각은 어느 청크에나 있어
#: 우연히 일치하고, 그러면 quote 검증이 통과 도장으로 전락한다.
MIN_QUOTE_CHARS = 20


def normalize_chunk_id(raw: str) -> str:
    """모델이 붙인 장식을 떼어낸다."""
    return raw.strip().strip(_ID_TRIM)


def _squash(text: str) -> str:
    """공백 차이만 무시한다. 단어를 바꾸는 변형은 그대로 걸린다."""
    return " ".join(text.split())


def quote_is_present(quote: str, body: str) -> bool:
    """인용이 본문에 실제로 있는가.

    정확 일치를 먼저 보고, 실패하면 공백만 정규화해 다시 본다. 모델이 개행을 공백으로
    바꾸는 것은 인용의 변형이 아니지만, 단어를 바꾸는 것은 변형이다.
    """
    if quote in body:
        return True
    return _squash(quote) in _squash(body)


def can_be_quoted(body: str) -> bool:
    """이 본문에서 현재 임계값을 통과하는 인용을 뽑을 수 있는가.

    라벨링 도구가 경고를 띄우는 데 쓴다. **경고는 하되 지정을 막지 않는다.**
    막으면 사람이 그런 청크를 피하게 되고, 그러면 MIN_QUOTE_CHARS 가 부적절해도
    영원히 드러나지 않는다. 이 임계값 자체가 측정 대상이다(ADR 0005).
    """
    return len(_squash(body)) >= MIN_QUOTE_CHARS


def _check(
    chunk_id: str, quote: str, bodies: Mapping[str, str], known_ids: Container[str] | None
) -> DropReason | None:
    """이 근거를 버려야 한다면 그 이유. 통과하면 None."""
    if chunk_id not in bodies:
        if known_ids is not None and chunk_id in known_ids:
            return DropReason.NOT_IN_CONTEXT
        return DropReason.UNKNOWN_CHUNK_ID
    if len(_squash(quote)) < MIN_QUOTE_CHARS:
        return DropReason.QUOTE_NOT_FOUND
    if not quote_is_present(quote, bodies[chunk_id]):
        return DropReason.QUOTE_NOT_FOUND
    return None


def verify_causes(
    raw_causes: Sequence[Mapping[str, Any]],
    context: Sequence[Chunk],
    known_ids: Container[str] | None = None,
) -> tuple[tuple[Cause, ...], tuple[DroppedEvidence, ...]]:
    """검증을 통과한 원인 후보와, 걸러진 근거 목록.

    근거가 하나도 남지 않은 원인 후보는 버린다. 근거 없는 주장은 이 프로젝트가
    내지 않기로 한 것이다.
    """
    bodies = {c.chunk_id: c.quotable for c in context}
    kept: list[Cause] = []
    dropped: list[DroppedEvidence] = []

    for raw in raw_causes:
        evidence: list[Evidence] = []
        for item in raw.get("evidence") or ():
            chunk_id = normalize_chunk_id(str(item.get("chunk_id", "")))
            quote = str(item.get("quote", ""))
            reason = _check(chunk_id, quote, bodies, known_ids)
            if reason is None:
                evidence.append(Evidence(chunk_id=chunk_id, quote=quote))
            else:
                dropped.append(DroppedEvidence(chunk_id=chunk_id, quote=quote, reason=reason))
        if not evidence:
            continue
        kept.append(
            Cause(
                description=str(raw.get("description", "")).strip(),
                evidence=tuple(evidence),
                confidence=Confidence.parse(str(raw.get("confidence", ""))),
            )
        )
    return tuple(kept), tuple(dropped)
