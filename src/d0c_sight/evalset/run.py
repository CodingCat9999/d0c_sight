"""평가 실행 — 폴백 차단, 캐싱, 불완전 표시.

세 가지가 이 모듈의 존재 이유다.

1. **모델 폴백을 끈다.** 다른 모델의 답을 같은 기준으로 비교하면 측정이 무의미해진다.
2. **캐싱한다.** 무료 티어 RPD 로는 이게 없으면 하루에 몇 번 못 돌린다.
3. **불완전한 실행을 완전한 것으로 집계하지 않는다.** 30문항 중 3개가 실패했는데
   27개만 보고 "87% 통과"라고 하면 틀린 숫자다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from d0c_sight.evalset.grade import CheckResult, grade_case
from d0c_sight.evalset.models import CaseType
from d0c_sight.llm.config import LlmConfig
from d0c_sight.llm.diagnose import DiagnosisError, diagnose
from d0c_sight.llm.prompt import PROMPT_VERSION
from d0c_sight.llm.protocol import ProviderError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from d0c_sight.domain.models import Chunk
    from d0c_sight.evalset.models import EvalCase, EvalSet
    from d0c_sight.llm.protocol import LlmProvider

log = logging.getLogger("d0c_sight.evalset")

#: 503 으로 실패한 문항을 몇 번까지 다시 돌릴 것인가. 폴백 대신 재실행으로 다룬다.
DEFAULT_MAX_RETRIES = 2


def cache_key(case: EvalCase, model: str, prompt_version: str, settings: Mapping[str, Any]) -> str:
    """(문항 + 프롬프트 버전 + 모델 + 설정) 이 같으면 같은 키.

    **프롬프트 버전이 빠지면 다른 프롬프트의 결과가 재사용되어 측정이 조용히
    오염된다.** 무엇을 쟀는지 모르는 숫자가 남는다.
    """
    payload = json.dumps(
        {
            "case_id": case.case_id,
            "question": case.question,
            "gold": sorted(g.chunk_id for g in case.gold_chunks),
            "model": model,
            "prompt_version": prompt_version,
            "settings": dict(sorted(settings.items())),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    checks: tuple[CheckResult, ...]
    insufficient_cause: str | None
    retries: int
    failed: bool
    failure_detail: str = ""
    from_cache: bool = False
    diagnosis: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return not self.failed and all(c.passed for c in self.checks)


@dataclass(frozen=True, slots=True)
class RunResult:
    """한 번의 평가 실행.

    `complete` 가 False 면 이 실행의 통과율은 **보고해서는 안 되는 숫자**다.
    """

    evalset_version: str
    corpus_version: str
    provider: str
    model: str
    prompt_version: str
    settings: dict[str, Any]
    cases: tuple[CaseResult, ...]
    started_at: str
    duration_s: float
    api_calls: int

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(c.case_id for c in self.cases if c.failed)

    @property
    def complete(self) -> bool:
        return not self.failed_case_ids

    @property
    def pass_rate(self) -> float | None:
        """불완전한 실행에서는 통과율을 계산하지 않는다. None 이 정답이다."""
        if not self.complete or not self.cases:
            return None
        return sum(1 for c in self.cases if c.passed_all) / len(self.cases)

    def check_totals(self) -> dict[str, tuple[int, int]]:
        """항목별 (통과, 전체). 총점만 보면 어느 항목이 뒤집혔는지 놓친다."""
        totals: dict[str, tuple[int, int]] = {}
        for case in self.cases:
            for check in case.checks:
                passed, total = totals.get(check.check_id.value, (0, 0))
                totals[check.check_id.value] = (passed + int(check.passed), total + 1)
        return totals

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["complete"] = self.complete
        data["failed_case_ids"] = list(self.failed_case_ids)
        data["pass_rate"] = self.pass_rate
        data["check_totals"] = {k: list(v) for k, v in self.check_totals().items()}
        return data


class ResultCache:
    """디스크 캐시. 키가 같으면 API 를 부르지 않는다."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._root / f"{key}.json"
        if not path.is_file():
            return None
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _context_for(case: EvalCase, chunks: Mapping[str, Chunk]) -> list[Chunk]:
    """정답 청크를 직접 주입한다. 검색 변수를 제거하고 생성 능력만 측정한다."""
    return [chunks[g.chunk_id] for g in case.gold_chunks if g.chunk_id in chunks]


def run_case(
    case: EvalCase,
    provider: LlmProvider,
    chunks: Mapping[str, Chunk],
    config: LlmConfig,
    cache: ResultCache | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[CaseResult, int]:
    """문항 하나를 실행하고 채점한다. (결과, 실제 API 호출 수)."""
    settings = {
        "thinking_budget": config.thinking_budget,
        "max_output_tokens": config.max_output_tokens,
    }
    key = cache_key(case, config.model, PROMPT_VERSION, settings)
    context = _context_for(case, chunks)

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return (
                CaseResult(
                    case_id=case.case_id,
                    checks=tuple(
                        CheckResult(**{**c, "check_id": c["check_id"]}) for c in hit["checks"]
                    ),
                    insufficient_cause=hit["insufficient_cause"],
                    retries=hit["retries"],
                    failed=hit["failed"],
                    failure_detail=hit.get("failure_detail", ""),
                    from_cache=True,
                    diagnosis=hit.get("diagnosis", {}),
                ),
                0,
            )

    # 폴백을 끈다. 같은 실행 안에서 모델이 바뀌면 비교가 불가능해진다.
    fixed = replace(config, fallback_models=())
    calls = 0
    last_error = ""
    for attempt in range(max_retries + 1):
        calls += 1
        try:
            result = diagnose(case.question, context, provider, fixed, chunks.keys())
        except (ProviderError, DiagnosisError) as exc:
            last_error = str(exc)
            log.warning("문항 실패 case=%s 시도=%d/%d", case.case_id, attempt + 1, max_retries + 1)
            continue

        checks = grade_case(case, result, chunks, context)
        payload = CaseResult(
            case_id=case.case_id,
            checks=checks,
            insufficient_cause=result.insufficient_cause,
            retries=attempt,
            failed=False,
            diagnosis=result.to_dict(),
        )
        if cache is not None:
            cache.put(key, asdict(payload))
        return payload, calls

    return (
        CaseResult(
            case_id=case.case_id,
            checks=(),
            insufficient_cause=None,
            retries=max_retries,
            failed=True,
            failure_detail=last_error,
        ),
        calls,
    )


def run_evalset(
    evalset: EvalSet,
    provider: LlmProvider,
    chunks: Mapping[str, Chunk],
    config: LlmConfig | None = None,
    cases: Sequence[EvalCase] | None = None,
    cache: ResultCache | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> RunResult:
    cfg = config or LlmConfig()
    targets = list(cases if cases is not None else evalset.cases)
    started = time.time()
    results: list[CaseResult] = []
    calls = 0
    for case in targets:
        result, used = run_case(case, provider, chunks, cfg, cache, max_retries)
        results.append(result)
        calls += used

    run = RunResult(
        evalset_version=evalset.version,
        corpus_version=evalset.corpus_version,
        provider=provider.provider_id,
        model=cfg.model,
        prompt_version=PROMPT_VERSION,
        settings={
            "thinking_budget": cfg.thinking_budget,
            "max_output_tokens": cfg.max_output_tokens,
            "max_retries": max_retries,
        },
        cases=tuple(results),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        duration_s=round(time.time() - started, 2),
        api_calls=calls,
    )
    if not run.complete:
        log.warning(
            "실행이 불완전하다. 실패 문항 %d개: %s — 통과율을 보고하지 마라",
            len(run.failed_case_ids),
            ", ".join(run.failed_case_ids),
        )
    return run


def no_answer_cases(evalset: EvalSet) -> tuple[EvalCase, ...]:
    return tuple(c for c in evalset.cases if c.case_type is CaseType.NO_ANSWER)
