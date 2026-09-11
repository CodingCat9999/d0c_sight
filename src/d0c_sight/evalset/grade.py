"""기계 채점 — 코드로 판정 가능한 항목 전부.

코드 채점은 공짜고 완벽히 재현되며 CI 에서 매번 돌 수 있다. judge 는 남는 것에만 쓴다.

모든 항목은 예/아니오다. 척도를 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from d0c_sight.evalset.models import CaseType
from d0c_sight.llm.verify import quote_is_present

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from d0c_sight.domain.models import Chunk, DiagnosisResult
    from d0c_sight.evalset.models import EvalCase


class CheckId(StrEnum):
    SCHEMA_VALID = "schema_valid"
    CITED_IDS_EXIST = "cited_ids_exist"
    QUOTES_VERIFIED = "quotes_verified"
    GOLD_CITED = "gold_cited"
    VERSION_MATCHES = "version_matches"
    NO_ANSWER_DECLINED = "no_answer_declined"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """한 항목의 판정. `passed` 는 예/아니오이며 중간값이 없다."""

    check_id: CheckId
    passed: bool
    detail: str = ""


def _cited(result: DiagnosisResult) -> list[tuple[str, str]]:
    return [(ev.chunk_id, ev.quote) for cause in result.diagnosis.causes for ev in cause.evidence]


def grade_case(
    case: EvalCase,
    result: DiagnosisResult,
    chunks: Mapping[str, Chunk],
    context: Sequence[Chunk],
) -> tuple[CheckResult, ...]:
    """한 문항의 기계 채점 결과.

    `no_answer` 문항에는 근거 관련 항목을 매기지 않는다. 근거가 없는 것이 정답이므로
    "정답 청크를 인용했는가"를 물으면 의미가 없다.
    """
    checks: list[CheckResult] = []
    cited = _cited(result)

    # 스키마는 파싱 성공 자체가 증거다. 파싱이 실패하면 여기 도달하지 않는다.
    checks.append(CheckResult(CheckId.SCHEMA_VALID, passed=True, detail="구조화 출력 파싱 성공"))

    if case.case_type is CaseType.NO_ANSWER:
        declined = result.diagnosis.insufficient_evidence and not result.diagnosis.causes
        checks.append(
            CheckResult(
                CheckId.NO_ANSWER_DECLINED,
                passed=declined,
                detail=(
                    f"insufficient_cause={result.insufficient_cause}"
                    if declined
                    else f"원인 {len(result.diagnosis.causes)}개를 지어냈다"
                ),
            )
        )
        return tuple(checks)

    unknown = [cid for cid, _ in cited if cid not in chunks]
    checks.append(
        CheckResult(
            CheckId.CITED_IDS_EXIST,
            passed=not unknown and bool(cited),
            detail="인용이 없다"
            if not cited
            else f"존재하지 않는 id: {unknown}"
            if unknown
            else "",
        )
    )

    bad_quotes = [
        cid
        for cid, quote in cited
        if cid in chunks and not quote_is_present(quote, chunks[cid].quotable)
    ]
    checks.append(
        CheckResult(
            CheckId.QUOTES_VERIFIED,
            passed=not bad_quotes and bool(cited),
            detail=f"본문에 없는 인용: {bad_quotes}" if bad_quotes else "",
        )
    )

    gold_ids = {g.chunk_id for g in case.gold_chunks}
    cited_ids = {cid for cid, _ in cited}
    hit = gold_ids & cited_ids
    checks.append(
        CheckResult(
            CheckId.GOLD_CITED,
            passed=bool(hit),
            detail=f"정답 {len(hit)}/{len(gold_ids)} 인용"
            + (f", 누락 {sorted(gold_ids - cited_ids)}" if gold_ids - cited_ids else ""),
        )
    )

    # 컨텍스트에 넣은 청크의 버전과 문항의 버전이 어긋나면 다른 버전의 답을 재는 셈이다.
    ctx_versions = {c.ref.version.full for c in context}
    matches = not ctx_versions or ctx_versions == {case.version}
    checks.append(
        CheckResult(
            CheckId.VERSION_MATCHES,
            passed=matches,
            detail="" if matches else f"문항 {case.version} vs 컨텍스트 {sorted(ctx_versions)}",
        )
    )
    return tuple(checks)
