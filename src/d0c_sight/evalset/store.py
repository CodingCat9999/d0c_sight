"""평가셋 파일 입출력.

DB 가 아니라 JSON 파일이다. 버전 관리가 되어야 하고 `git diff` 로 무엇이 바뀌었는지
볼 수 있어야 한다. 평가셋 파일 자체가 이 프로젝트의 산출물이기도 하다.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from d0c_sight.evalset.models import (
    CaseType,
    EvalCase,
    EvalSet,
    GoldChunk,
    Provenance,
    SearchAxis,
    Split,
    Trap,
)

if TYPE_CHECKING:
    from pathlib import Path


def _case_from(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        case_id=row["case_id"],
        question=row["question"],
        version=row["version"],
        case_type=CaseType(row["case_type"]),
        search_axis=SearchAxis(row["search_axis"]),
        gold_chunks=tuple(GoldChunk(**g) for g in row.get("gold_chunks", ())),
        split=Split(row["split"]),
        provenance=Provenance(row["provenance"]),
        rationale=row["rationale"],
        source_url=row.get("source_url"),
        traps=tuple(Trap(t) for t in row.get("traps", ())),
        judge_items=tuple(row.get("judge_items", ())),
        needs_review=bool(row.get("needs_review", False)),
        review_note=row.get("review_note", ""),
    )


def load(path: Path) -> EvalSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EvalSet(
        version=raw["version"],
        created=raw["created"],
        corpus_version=raw["corpus_version"],
        cases=tuple(_case_from(r) for r in raw.get("cases", ())),
    )


def save(evalset: EvalSet, path: Path) -> None:
    """읽기 좋게 저장한다. diff 가 사람에게 읽혀야 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(evalset)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
