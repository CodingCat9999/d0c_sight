"""평가셋 무결성 — 라벨이 조용히 썩는 것을 막는다.

두 가지를 본다.

1. 정답 청크가 여전히 존재하는가
2. **존재하지만 내용이 바뀌지 않았는가**

두 번째가 핵심이다. 18.4→18.6 실측에서 chunk_id 는 100% 유지됐지만 58개(0.35%)는
본문이 바뀌었다. 존재 확인만 하면 이 경우를 놓치고, 정답 라벨이 조용히 틀려진다.

**자동으로 무효화하지 않는다.** 문서가 개정돼도 여전히 정답일 수 있고 아닐 수도 있다.
기계가 판단할 수 없는 영역이므로 "재검토 필요"로 표시하고 사람에게 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from d0c_sight.evalset.models import EvalSet, content_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from d0c_sight.domain.models import Chunk


class IssueKind(StrEnum):
    MISSING = "missing"
    CONTENT_CHANGED = "content_changed"


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    case_id: str
    chunk_id: str
    kind: IssueKind
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.case_id} / {self.chunk_id} — {self.detail}"


def check(evalset: EvalSet, chunks: Mapping[str, Chunk]) -> tuple[IntegrityIssue, ...]:
    """문제가 없으면 빈 튜플."""
    issues: list[IntegrityIssue] = []
    for case in evalset.cases:
        for gold in case.gold_chunks:
            chunk = chunks.get(gold.chunk_id)
            if chunk is None:
                issues.append(
                    IntegrityIssue(
                        case_id=case.case_id,
                        chunk_id=gold.chunk_id,
                        kind=IssueKind.MISSING,
                        detail="코퍼스에 없다. 재인덱싱으로 사라졌을 수 있다",
                    )
                )
                continue
            actual = content_hash(chunk.quotable)
            if actual != gold.content_sha256:
                issues.append(
                    IntegrityIssue(
                        case_id=case.case_id,
                        chunk_id=gold.chunk_id,
                        kind=IssueKind.CONTENT_CHANGED,
                        detail=f"본문이 바뀌었다 ({gold.content_sha256} → {actual})",
                    )
                )
    return tuple(issues)


def mark_for_review(evalset: EvalSet, issues: tuple[IntegrityIssue, ...]) -> EvalSet:
    """문제가 있는 문항을 "재검토 필요"로 표시한다. 폐기하지 않는다.

    사람이 보고 정답 유지 / 라벨 수정 / 문항 폐기 중 하나를 고른다.
    """
    by_case: dict[str, list[IntegrityIssue]] = {}
    for issue in issues:
        by_case.setdefault(issue.case_id, []).append(issue)
    if not by_case:
        return evalset
    cases = tuple(
        replace(
            case,
            needs_review=True,
            review_note="; ".join(str(i) for i in by_case[case.case_id]),
        )
        if case.case_id in by_case
        else case
        for case in evalset.cases
    )
    return replace(evalset, cases=cases)
