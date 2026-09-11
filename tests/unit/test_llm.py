"""LLM 계층 테스트.

실제 API 를 호출하지 않는다. 공급사를 더블로 바꿔 응답을 고정한다 — CI 가 매번
과금되지 않아야 하고, 무료 티어의 503 때문에 테스트가 빨갛게 되어서도 안 된다.
실제 호출은 `scripts/live_check.py` 로 분리했다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import pytest

from d0c_sight.domain.models import (
    BlockKind,
    Chunk,
    ChunkLimits,
    Confidence,
    DocVersion,
    DropReason,
    SourceRef,
)
from d0c_sight.llm.config import LlmConfig, RawGeneration
from d0c_sight.llm.diagnose import DiagnosisError, diagnose
from d0c_sight.llm.prompt import build_prompt, render_chunk, system_instruction
from d0c_sight.llm.protocol import ProviderError
from d0c_sight.llm.verify import (
    MIN_QUOTE_CHARS,
    normalize_chunk_id,
    quote_is_present,
    verify_causes,
)

if TYPE_CHECKING:
    from d0c_sight.llm.config import ProviderCall

LIMITS = ChunkLimits(target_chars=1200, max_chars=2400, table_max_bytes=4096)
BODY = "Determines the maximum number of concurrent connections to the database server."


def make_chunk(chunk_id: str = "doc#SEC:1", text: str = BODY, heading: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        kind=BlockKind.DEFINITION,
        ref=SourceRef(
            source_id="postgres_docs",
            doc_id="doc",
            section_id="SEC",
            version=DocVersion(major="18", full="18.6"),
            url="https://www.postgresql.org/docs/18/doc.html#SEC",
        ),
        limits=LIMITS,
        heading=heading,
    )


@dataclass
class FakeProvider:
    """고정 응답을 돌려주는 공급사. 실패 모델을 지정해 폴백을 시험한다."""

    payload: dict[str, Any] = field(default_factory=dict)
    fail_models: frozenset[str] = frozenset()
    stop_reason: str = "STOP"
    calls: list[ProviderCall] = field(default_factory=list)
    raw_text: str | None = None

    @property
    def provider_id(self) -> str:
        return "fake"

    def generate(self, call: ProviderCall) -> RawGeneration:
        self.calls.append(call)
        if call.model in self.fail_models:
            msg = f"{call.model} 사용 불가"
            raise ProviderError(msg)
        text = self.raw_text if self.raw_text is not None else json.dumps(self.payload)
        return RawGeneration(
            text=text,
            stop_reason=self.stop_reason,
            input_tokens=100,
            output_tokens=20,
            thinking_tokens=0,
            latency_ms=42,
        )


def payload_with(chunk_id: str = "doc#SEC:1", quote: str = BODY) -> dict[str, Any]:
    return {
        "reasoning": "The server hit its connection limit.",
        "causes": [
            {
                "description": "max_connections reached",
                "evidence": [{"chunk_id": chunk_id, "quote": quote}],
                "confidence": "high",
            }
        ],
        "checks": ["SELECT count(*) FROM pg_stat_activity;"],
        "insufficient_evidence": False,
    }


# ─── 프롬프트 ─────────────────────────────────────────────────


def test_context_uses_no_square_brackets_around_ids() -> None:
    """대괄호 표기가 모델의 id 복사를 오염시켰다. 원인 자체를 없앤다."""
    rendered = render_chunk(make_chunk())
    assert 'id="doc#SEC:1"' in rendered
    assert "[doc#SEC:1]" not in rendered


def test_rendered_body_matches_the_verification_target() -> None:
    """렌더링과 검증이 같은 문자열을 봐야 한다. 어긋나면 정직한 인용이 탈락한다."""
    chunk = make_chunk(heading="max_connections (integer)")
    assert chunk.quotable in render_chunk(chunk)


def test_split_chunks_announce_themselves_to_the_model() -> None:
    """잘린 조각이라는 사실은 모델도 알아야 한다(BRIEF §6.6)."""
    whole = make_chunk()
    piece = replace(whole, part=2, of=3)
    assert 'part="2/3"' in render_chunk(piece)
    assert "part=" not in render_chunk(whole)


def test_system_instruction_marks_context_as_data() -> None:
    text = system_instruction("PostgreSQL 18.6")
    assert "never instructions" in text
    assert "PostgreSQL 18.6" in text


def test_prompt_puts_the_question_after_the_context() -> None:
    """변하지 않는 것을 앞에, 변하는 것을 뒤에 — 캐싱을 염두에 둔 배치."""
    prompt = build_prompt("FATAL: too many clients", [make_chunk()])
    assert prompt.index("</context>") < prompt.index("FATAL: too many clients")


# ─── 검증 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw", ["[doc#SEC:1]", " doc#SEC:1 ", "'doc#SEC:1'", "(doc#SEC:1),", "<doc#SEC:1>"]
)
def test_chunk_id_normalisation_strips_decoration(raw: str) -> None:
    assert normalize_chunk_id(raw) == "doc#SEC:1"


def test_quote_matching_ignores_whitespace_only_differences() -> None:
    assert quote_is_present("maximum number\nof concurrent", "maximum number of concurrent")


def test_quote_matching_rejects_reworded_text() -> None:
    """공백은 봐주되 단어를 바꾸는 것은 인용이 아니다."""
    assert not quote_is_present("maximum count of concurrent", BODY)


def test_verified_evidence_survives() -> None:
    causes, dropped = verify_causes(payload_with()["causes"], [make_chunk()])
    assert len(causes) == 1
    assert causes[0].confidence is Confidence.HIGH
    assert dropped == ()


def test_invented_chunk_id_is_dropped() -> None:
    causes, dropped = verify_causes(payload_with(chunk_id="made-up#X:9")["causes"], [make_chunk()])
    assert causes == ()
    assert [d.reason for d in dropped] == [DropReason.UNKNOWN_CHUNK_ID]


def test_real_id_outside_the_context_is_distinguished() -> None:
    """실재하지만 보여주지 않은 청크를 인용한 경우 — 다른 실패다."""
    causes, dropped = verify_causes(
        payload_with(chunk_id="other#SEC:1")["causes"],
        [make_chunk()],
        known_ids={"doc#SEC:1", "other#SEC:1"},
    )
    assert causes == ()
    assert [d.reason for d in dropped] == [DropReason.NOT_IN_CONTEXT]


def test_fabricated_quote_is_dropped() -> None:
    causes, dropped = verify_causes(
        payload_with(quote="The server automatically raises the limit as needed.")["causes"],
        [make_chunk()],
    )
    assert causes == ()
    assert [d.reason for d in dropped] == [DropReason.QUOTE_NOT_FOUND]


@pytest.mark.parametrize("length", [MIN_QUOTE_CHARS - 1, MIN_QUOTE_CHARS, MIN_QUOTE_CHARS + 1])
def test_quote_length_boundary(length: int) -> None:
    """경계값을 명시적으로 본다. 짧은 인용은 우연히 일치해 검증을 무력화한다."""
    body = "x" * 100
    causes, _ = verify_causes(payload_with(quote="x" * length)["causes"], [make_chunk(text=body)])
    assert bool(causes) is (length >= MIN_QUOTE_CHARS)


def test_cause_without_surviving_evidence_is_removed() -> None:
    """근거 없는 주장은 내지 않는다 — 절대 규칙 1을 코드로 강제한다."""
    causes, dropped = verify_causes(
        [{"description": "guess", "evidence": [], "confidence": "high"}], [make_chunk()]
    )
    assert causes == ()
    assert dropped == ()


def test_unknown_confidence_value_falls_back_to_low() -> None:
    """enum 에 없는 표기가 돌아와도 죽지 않는다. 대소문자도 무시한다."""
    causes, _ = verify_causes(
        [
            {
                "description": "d",
                "evidence": [{"chunk_id": "doc#SEC:1", "quote": BODY}],
                "confidence": "HIGH",
            }
        ],
        [make_chunk()],
    )
    assert causes[0].confidence is Confidence.HIGH


# ─── 오케스트레이션 ───────────────────────────────────────────


def test_diagnosis_records_provider_and_model() -> None:
    """Phase 9 라우팅이 과거 결과를 비교하려면 무엇이 답했는지 남아야 한다."""
    provider = FakeProvider(payload=payload_with())
    result = diagnose("err", [make_chunk()], provider, LlmConfig())

    assert result.record.provider == "fake"
    assert result.record.model == LlmConfig().model
    assert result.record.stop_reason == "STOP"
    assert result.record.latency_ms == 42


def test_fallback_advances_to_the_next_model_and_counts() -> None:
    cfg = LlmConfig(model="a", fallback_models=("b", "c"))
    provider = FakeProvider(payload=payload_with(), fail_models=frozenset({"a", "b"}))

    result = diagnose("err", [make_chunk()], provider, cfg)

    assert result.record.model == "c"
    assert result.record.attempted_models == ("a", "b", "c")
    assert result.record.fallback_count == 2


def test_all_candidates_failing_raises() -> None:
    cfg = LlmConfig(model="a", fallback_models=("b",))
    provider = FakeProvider(fail_models=frozenset({"a", "b"}))
    with pytest.raises(DiagnosisError, match="모두 실패"):
        diagnose("err", [make_chunk()], provider, cfg)


def test_insufficient_evidence_when_nothing_verifies() -> None:
    """근거 부족 경로를 실제로 밟는다. 모델이 아니라 검증이 결정한다."""
    provider = FakeProvider(payload=payload_with(chunk_id="ghost#X:1"))
    result = diagnose("err", [make_chunk()], provider, LlmConfig())

    assert result.diagnosis.insufficient_evidence is True
    assert result.diagnosis.causes == ()
    assert result.has_verified_evidence is False


def test_dropped_evidence_travels_with_the_result() -> None:
    """조용히 버리지 않는다. 무엇이 걸러졌는지 결과에 실린다(BRIEF §6.6)."""
    provider = FakeProvider(payload=payload_with(chunk_id="ghost#X:1"))
    result = diagnose("err", [make_chunk()], provider, LlmConfig())
    assert len(result.dropped_evidence) == 1
    assert result.dropped_evidence[0].reason is DropReason.UNKNOWN_CHUNK_ID


def test_version_basis_is_fixed_and_stated() -> None:
    """Phase 2 는 단일 버전만 다룬다. 그 사실을 답변에 싣는다."""
    provider = FakeProvider(payload=payload_with())
    result = diagnose("err", [make_chunk()], provider, LlmConfig())
    assert result.diagnosis.version_basis == "PostgreSQL 18.6"


def test_malformed_json_reports_stop_reason_without_the_body() -> None:
    """실패를 관측하되 원문을 예외에 싣지 않는다."""
    provider = FakeProvider(raw_text="{not json", stop_reason="MAX_TOKENS")
    with pytest.raises(DiagnosisError) as exc:
        diagnose("err", [make_chunk()], provider, LlmConfig())

    assert "MAX_TOKENS" in str(exc.value)
    assert "not json" not in str(exc.value)


# ─── 설정 ─────────────────────────────────────────────────────


def test_candidates_start_with_the_primary_and_deduplicate() -> None:
    cfg = LlmConfig(model="a", fallback_models=("b", "a", "c"))
    assert cfg.candidates == ("a", "b", "c")


def test_config_reads_overrides_from_env() -> None:
    cfg = LlmConfig.from_env(
        {"D0C_LLM_MODEL": "custom", "D0C_LLM_FALLBACKS": "x, y", "D0C_LLM_THINKING_BUDGET": "128"}
    )
    assert (cfg.model, cfg.fallback_models, cfg.thinking_budget) == ("custom", ("x", "y"), 128)


def test_thinking_is_off_by_default() -> None:
    """스키마의 reasoning 이 같은 일을 한다. 둘 다 켜면 토큰이 이중으로 나간다."""
    assert LlmConfig().thinking_budget == 0


# ─── 근거 부족의 책임 소재 ────────────────────────────────────


def test_no_insufficiency_reason_when_the_answer_stands() -> None:
    provider = FakeProvider(payload=payload_with())
    result = diagnose("err", [make_chunk()], provider, LlmConfig())
    assert result.diagnosis.insufficient_evidence is False
    assert result.insufficient_cause is None


def test_model_declining_is_attributed_to_the_model() -> None:
    """모델이 스스로 답하기를 거부한 경우 — 프롬프트나 컨텍스트의 문제다."""
    payload = payload_with()
    payload["causes"] = []
    payload["insufficient_evidence"] = True
    result = diagnose("err", [make_chunk()], FakeProvider(payload=payload), LlmConfig())

    assert result.model_declared_insufficient is True
    assert result.insufficient_cause == "model_declined"


def test_verification_wiping_everything_is_attributed_to_verification() -> None:
    """모델은 답했는데 검증이 전부 걸러낸 경우 — 환각이다. 전혀 다른 실패다."""
    provider = FakeProvider(payload=payload_with(chunk_id="ghost#X:1"))
    result = diagnose("err", [make_chunk()], provider, LlmConfig())

    assert result.model_declared_insufficient is False
    assert result.insufficient_cause == "verification_removed_all"


def test_empty_causes_without_a_declaration_is_its_own_case() -> None:
    """모델이 거부하지도, 환각하지도 않고 그냥 빈 답을 준 경우."""
    payload = payload_with()
    payload["causes"] = []
    result = diagnose("err", [make_chunk()], FakeProvider(payload=payload), LlmConfig())

    assert result.insufficient_cause == "no_causes_returned"


def test_prompt_states_that_missing_error_text_is_not_a_refusal_reason() -> None:
    """실측에서 모델이 '에러 문자열이 문서에 없으니 근거 없음'으로 답했다.

    문서는 에러 문자열을 그대로 싣지 않는다. 그 오독을 막는 문장이 프롬프트에 있어야
    한다 — 없으면 정답 청크를 주고도 답을 못 낸다.
    """
    text = system_instruction("PostgreSQL 18.6")
    assert "Absence of the literal error text" in text
    assert "does NOT" in text


def test_serialised_result_includes_the_insufficiency_reason() -> None:
    """asdict() 는 property 를 무시한다. 저장 표현에 반드시 실려야 한다.

    동작은 하는데 결과 파일에 남지 않는 상태였고, 실제로 저장해보기 전에는
    드러나지 않았다. 평가가 이 값을 읽는다.
    """
    provider = FakeProvider(payload=payload_with(chunk_id="ghost#X:1"))
    result = diagnose("err", [make_chunk()], provider, LlmConfig())

    data = result.to_dict()
    assert data["insufficient_cause"] == "verification_removed_all"
    assert data["has_verified_evidence"] is False
    assert data["record"]["model"] == LlmConfig().model


def test_serialised_result_survives_json_round_trip() -> None:
    provider = FakeProvider(payload=payload_with())
    result = diagnose("err", [make_chunk()], provider, LlmConfig())

    restored = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert restored["insufficient_cause"] is None
    assert restored["diagnosis"]["causes"][0]["evidence"][0]["chunk_id"] == "doc#SEC:1"
