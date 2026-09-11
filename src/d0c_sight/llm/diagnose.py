"""진단 오케스트레이션 — 호출, 파싱, 검증, 기록.

이 Phase 에서 하는 일은 **어떤 실패가 실제로 일어나는지 관측하는 것**이다. 정교한
재시도 정책을 만들지 않는다. SDK 가 이미 5회 지수 백오프로 재시도하며 429 도 그
대상이다. 그 위에 얹은 것은 모델 폴백 하나뿐이고, 그것도 가용성 때문이지 품질
때문이 아니다(ADR 0006).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from d0c_sight.domain.models import (
    Diagnosis,
    DiagnosisResult,
    GenerationRecord,
)
from d0c_sight.llm.config import LlmConfig, ProviderCall
from d0c_sight.llm.prompt import RESPONSE_SCHEMA, build_prompt, system_instruction
from d0c_sight.llm.protocol import ProviderError
from d0c_sight.llm.verify import verify_causes

if TYPE_CHECKING:
    from collections.abc import Container, Sequence

    from d0c_sight.domain.models import Chunk
    from d0c_sight.llm.config import RawGeneration
    from d0c_sight.llm.protocol import LlmProvider

log = logging.getLogger("d0c_sight.llm")

#: Phase 2 는 단일 버전만 다룬다. 질문에 버전이 없을 때의 동작은 아직 미결이며,
#: 여기서는 고정 버전을 답변에 명시하는 것으로 우회한다.
VERSION_BASIS = "PostgreSQL 18.6"


class DiagnosisError(RuntimeError):
    """모든 후보 모델이 실패했다."""


def _parse(raw: RawGeneration) -> dict[str, Any]:
    try:
        payload = json.loads(raw.text)
    except json.JSONDecodeError as exc:
        # 원문을 예외에 싣지 않는다. 길이와 위치만으로 진단에 충분하다.
        msg = (
            f"JSON 파싱 실패 (stop_reason={raw.stop_reason}, "
            f"길이={len(raw.text)}자, 위치={exc.pos})"
        )
        raise DiagnosisError(msg) from None
    if not isinstance(payload, dict):
        msg = f"최상위가 객체가 아니다: {type(payload).__name__}"
        raise DiagnosisError(msg)
    return payload


def _insufficient_cause(insufficient: bool, declared: bool, dropped: Sequence[object]) -> str:
    """로그용 사유 라벨. DiagnosisResult.insufficient_cause 와 같은 구분이다."""
    if not insufficient:
        return "-"
    if declared:
        return "model_declined"
    return "verification_removed_all" if dropped else "no_causes_returned"


def diagnose(
    error_message: str,
    chunks: Sequence[Chunk],
    provider: LlmProvider,
    config: LlmConfig | None = None,
    known_ids: Container[str] | None = None,
) -> DiagnosisResult:
    """에러 메시지 하나를 진단한다. 검색은 하지 않는다 — 청크는 호출자가 고른다."""
    cfg = config or LlmConfig()
    call_base = ProviderCall(
        model="",
        system_instruction=system_instruction(VERSION_BASIS),
        prompt=build_prompt(error_message, chunks),
        response_schema=RESPONSE_SCHEMA,
        thinking_budget=cfg.thinking_budget,
        max_output_tokens=cfg.max_output_tokens,
    )

    attempted: list[str] = []
    failures: list[str] = []
    for model in cfg.candidates:
        attempted.append(model)
        try:
            raw = provider.generate(
                ProviderCall(
                    model=model,
                    system_instruction=call_base.system_instruction,
                    prompt=call_base.prompt,
                    response_schema=call_base.response_schema,
                    thinking_budget=call_base.thinking_budget,
                    max_output_tokens=call_base.max_output_tokens,
                )
            )
        except ProviderError as exc:
            failures.append(f"{model}: {exc}")
            log.warning(
                "모델 호출 실패, 폴백 진행 model=%s 시도=%d/%d",
                model,
                len(attempted),
                len(cfg.candidates),
            )
            continue

        payload = _parse(raw)
        causes, dropped = verify_causes(payload.get("causes") or (), chunks, known_ids)
        declared = bool(payload.get("insufficient_evidence"))
        insufficient = declared or not causes

        record = GenerationRecord(
            provider=provider.provider_id,
            model=model,
            attempted_models=tuple(attempted),
            stop_reason=raw.stop_reason,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            thinking_tokens=raw.thinking_tokens,
            latency_ms=raw.latency_ms,
        )
        log.info(
            "진단 완료 provider=%s model=%s stop=%s in=%d out=%d think=%d %dms "
            "폴백=%d 원인=%d 근거탈락=%d 근거부족사유=%s",
            record.provider,
            record.model,
            record.stop_reason,
            record.input_tokens,
            record.output_tokens,
            record.thinking_tokens,
            record.latency_ms,
            record.fallback_count,
            len(causes),
            len(dropped),
            _insufficient_cause(insufficient, declared, dropped),
        )
        return DiagnosisResult(
            diagnosis=Diagnosis(
                reasoning=str(payload.get("reasoning", "")).strip(),
                causes=causes,
                checks=tuple(str(c) for c in (payload.get("checks") or ())),
                version_basis=VERSION_BASIS,
                insufficient_evidence=insufficient,
            ),
            record=record,
            dropped_evidence=dropped,
            model_declared_insufficient=declared,
        )

    msg = f"후보 {len(cfg.candidates)}개 모두 실패: " + " | ".join(failures)
    raise DiagnosisError(msg)
