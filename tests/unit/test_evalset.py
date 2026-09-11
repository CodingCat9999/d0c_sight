"""평가셋 테스트.

judge 와 실제 API 는 호출하지 않는다. 공급사를 더블로 고정한다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from d0c_sight.evalset import integrity, retrieval, store
from d0c_sight.evalset.grade import CheckId, grade_case
from d0c_sight.evalset.models import (
    CaseType,
    EvalCase,
    EvalSet,
    GoldChunk,
    Provenance,
    SearchAxis,
    Split,
    content_hash,
)
from d0c_sight.evalset.run import (
    ResultCache,
    cache_key,
    run_case,
    run_evalset,
)
from d0c_sight.llm.config import LlmConfig
from d0c_sight.llm.diagnose import diagnose
from d0c_sight.llm.prompt import PROMPT_VERSION
from tests.unit.test_llm import FakeProvider, make_chunk, payload_with

if TYPE_CHECKING:
    from d0c_sight.domain.models import Chunk

GOLD_ID = "doc#SEC:1"


def gold_case(**over: Any) -> EvalCase:
    base = EvalCase(
        case_id="c1",
        question="too many clients",
        version="18.6",
        case_type=CaseType.ERROR,
        search_axis=SearchAxis.BOTH,
        gold_chunks=(GoldChunk(GOLD_ID, content_hash(make_chunk().quotable)),),
        split=Split.IMPROVE,
        provenance=Provenance.DOCUMENTATION,
        rationale="연결 수 초과의 기본 경로",
    )
    return replace(base, **over) if over else base


def no_answer_case() -> EvalCase:
    return EvalCase(
        case_id="n1",
        question="PostgreSQL 이 내 회사 VPN 설정을 어떻게 바꾸나요?",
        version="18.6",
        case_type=CaseType.NO_ANSWER,
        search_axis=SearchAxis.SEMANTIC_ONLY,
        gold_chunks=(),
        split=Split.IMPROVE,
        provenance=Provenance.DOCUMENTATION,
        rationale="문서에 없는 것을 물었을 때 지어내지 않는지 본다",
    )


def an_evalset(*cases: EvalCase) -> EvalSet:
    return EvalSet(version="v1", created="2026-09-11", corpus_version="18.6", cases=cases)


# ─── 모델 불변식 ──────────────────────────────────────────────


def test_no_answer_case_rejects_gold_chunks() -> None:
    """답이 없어야 정상인 문항에 정답이 있으면 문항이 잘못 쓰인 것이다."""
    with pytest.raises(ValueError, match="no_answer"):
        EvalCase(
            case_id="x",
            question="q",
            version="18.6",
            case_type=CaseType.NO_ANSWER,
            search_axis=SearchAxis.BOTH,
            gold_chunks=(GoldChunk("a", "b"),),
            split=Split.IMPROVE,
            provenance=Provenance.DOCUMENTATION,
            rationale="r",
        )


def test_answerable_case_requires_gold_chunks() -> None:
    with pytest.raises(ValueError, match="정답 청크가 없다"):
        gold_case(gold_chunks=())


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="중복"):
        an_evalset(gold_case(), gold_case())


def test_split_separates_holdout() -> None:
    """홀드아웃이 개선용과 섞이면 과적합을 감지할 수 없다."""
    es = an_evalset(gold_case(), gold_case(case_id="c2", split=Split.HOLDOUT))
    assert [c.case_id for c in es.by_split(Split.IMPROVE)] == ["c1"]
    assert [c.case_id for c in es.by_split(Split.HOLDOUT)] == ["c2"]


# ─── 무결성 ───────────────────────────────────────────────────


def test_integrity_passes_when_corpus_matches() -> None:
    chunks = {GOLD_ID: make_chunk()}
    assert integrity.check(an_evalset(gold_case()), chunks) == ()


def test_missing_gold_chunk_is_reported() -> None:
    """정답 청크가 사라진 평가셋은 조용히 통과하면 안 된다."""
    issues = integrity.check(an_evalset(gold_case()), {})
    assert len(issues) == 1
    assert issues[0].kind is integrity.IssueKind.MISSING


def test_changed_gold_content_is_reported() -> None:
    """id 는 살아 있는데 본문이 바뀐 경우. 존재 확인만으로는 잡히지 않는다."""
    chunks = {GOLD_ID: make_chunk(text="문서가 개정되어 내용이 달라졌다")}
    issues = integrity.check(an_evalset(gold_case()), chunks)

    assert len(issues) == 1
    assert issues[0].kind is integrity.IssueKind.CONTENT_CHANGED


def test_integrity_does_not_auto_invalidate() -> None:
    """자동으로 폐기하지 않는다. 개정된 문서가 여전히 정답일 수 있다."""
    es = an_evalset(gold_case())
    issues = integrity.check(es, {GOLD_ID: make_chunk(text="바뀐 본문")})
    marked = integrity.mark_for_review(es, issues)

    assert len(marked.cases) == 1, "문항이 사라지면 안 된다"
    assert marked.cases[0].needs_review is True
    assert "content_changed" in marked.cases[0].review_note


def test_clean_evalset_is_returned_unchanged() -> None:
    es = an_evalset(gold_case())
    assert integrity.mark_for_review(es, ()) is es


# ─── 저장 ─────────────────────────────────────────────────────


def test_evalset_round_trips_through_json(tmp_path: Path) -> None:
    es = an_evalset(gold_case(), no_answer_case())
    path = tmp_path / "eval.json"
    store.save(es, path)
    assert store.load(path) == es


def test_saved_file_is_readable_as_a_diff(tmp_path: Path) -> None:
    """git diff 로 무엇이 바뀌었는지 보여야 한다."""
    path = tmp_path / "eval.json"
    store.save(an_evalset(gold_case()), path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '      "case_id": "c1"' in text


# ─── 기계 채점 ────────────────────────────────────────────────


def _graded(case: EvalCase, payload: dict[str, object]) -> dict[str, bool]:
    chunk = make_chunk()
    chunks = {GOLD_ID: chunk}
    result = diagnose(case.question, [chunk], FakeProvider(payload=payload), LlmConfig())
    return {c.check_id.value: c.passed for c in grade_case(case, result, chunks, [chunk])}


def test_correct_answer_passes_every_machine_check() -> None:
    graded = _graded(gold_case(), payload_with())
    assert all(graded.values()), graded


def test_missing_gold_citation_fails_that_check_only() -> None:
    """정답을 인용하지 않았다 — 다른 항목은 통과할 수 있어야 한다."""
    other = make_chunk(chunk_id="doc#SEC:1")
    graded = _graded(gold_case(gold_chunks=(GoldChunk("elsewhere#X:1", "h"),)), payload_with())

    assert graded[CheckId.GOLD_CITED.value] is False
    assert graded[CheckId.QUOTES_VERIFIED.value] is True
    assert other.chunk_id == GOLD_ID


def test_fabricated_quote_fails_the_quote_check() -> None:
    graded = _graded(
        gold_case(), payload_with(quote="이 문장은 청크에 없다. 완전히 지어낸 것이다.")
    )
    assert graded[CheckId.QUOTES_VERIFIED.value] is False


def test_no_answer_case_passes_when_the_model_declines() -> None:
    payload = payload_with()
    payload["causes"] = []
    payload["insufficient_evidence"] = True
    graded = _graded(no_answer_case(), payload)
    assert graded[CheckId.NO_ANSWER_DECLINED.value] is True


def test_no_answer_case_fails_when_the_model_invents_an_answer() -> None:
    """문서에 없는 것을 물었는데 답을 지어내면 실패다. 이 경로가 핵심이다."""
    graded = _graded(no_answer_case(), payload_with())
    assert graded[CheckId.NO_ANSWER_DECLINED.value] is False


def test_no_answer_case_skips_evidence_checks() -> None:
    """근거가 없는 것이 정답이므로 근거 항목을 매기면 의미가 없다."""
    payload = payload_with()
    payload["causes"] = []
    payload["insufficient_evidence"] = True
    graded = _graded(no_answer_case(), payload)
    assert CheckId.GOLD_CITED.value not in graded


# ─── 검색 지표 ────────────────────────────────────────────────


def test_recall_counts_only_the_top_k() -> None:
    s = retrieval.score(["a", "b", "c", "d"], {"a", "d"}, k=2)
    assert s.recall_at_k == 0.5
    assert s.k == 2


@pytest.mark.parametrize("rank,expected", [(1, 1.0), (2, 0.5), (4, 0.25)])
def test_reciprocal_rank_uses_the_first_hit(rank: int, expected: float) -> None:
    retrieved = [f"x{i}" for i in range(10)]
    retrieved[rank - 1] = "gold"
    assert retrieval.score(retrieved, {"gold"}, k=5).reciprocal_rank == expected


def test_no_answer_case_does_not_drag_down_recall() -> None:
    """찾을 것이 없으면 '다 찾았다'가 맞다."""
    s = retrieval.score(["a"], set(), k=5)
    assert s.recall_at_k == 1.0
    assert s.gold_total == 0


def test_k_is_recorded_with_the_score() -> None:
    """k 를 바꿔도 과거 결과와 비교할 수 있어야 한다."""
    assert retrieval.score(["a"], {"a"}, k=3).k == 3


# ─── 캐시 키 ──────────────────────────────────────────────────

SETTINGS = {"thinking_budget": 0, "max_output_tokens": 4096}


def test_same_inputs_produce_the_same_key() -> None:
    a = cache_key(gold_case(), "m", PROMPT_VERSION, SETTINGS)
    b = cache_key(gold_case(), "m", PROMPT_VERSION, SETTINGS)
    assert a == b


def test_changing_the_prompt_version_changes_the_key() -> None:
    """프롬프트가 바뀌면 캐시가 무효가 되어야 한다.

    이것이 없으면 다른 프롬프트의 결과가 재사용되고, 무엇을 쟀는지 모르는 숫자가
    남는다. 측정이 조용히 오염되는 경로다.
    """
    old = cache_key(gold_case(), "m", "p1", SETTINGS)
    new = cache_key(gold_case(), "m", "p2", SETTINGS)
    assert old != new


def test_changing_the_model_changes_the_key() -> None:
    assert cache_key(gold_case(), "a", PROMPT_VERSION, SETTINGS) != cache_key(
        gold_case(), "b", PROMPT_VERSION, SETTINGS
    )


def test_changing_settings_changes_the_key() -> None:
    assert cache_key(gold_case(), "m", PROMPT_VERSION, SETTINGS) != cache_key(
        gold_case(), "m", PROMPT_VERSION, {**SETTINGS, "thinking_budget": 512}
    )


def test_changing_the_question_changes_the_key() -> None:
    assert cache_key(gold_case(), "m", PROMPT_VERSION, SETTINGS) != cache_key(
        gold_case(question="다른 질문"), "m", PROMPT_VERSION, SETTINGS
    )


# ─── 실행 ─────────────────────────────────────────────────────


def _chunks() -> dict[str, Chunk]:
    return {GOLD_ID: make_chunk()}


def test_run_disables_model_fallback() -> None:
    """평가 중 폴백이 일어나면 다른 모델의 답을 같은 기준으로 비교하게 된다."""
    provider = FakeProvider(payload=payload_with(), fail_models=frozenset({"primary"}))
    cfg = LlmConfig(model="primary", fallback_models=("secondary",))

    result, _ = run_case(gold_case(), provider, _chunks(), cfg, max_retries=0)

    assert result.failed is True, "폴백이 켜져 있었다면 secondary 로 성공했을 것이다"
    assert [c.model for c in provider.calls] == ["primary"]


def test_transient_failure_is_retried_and_counted() -> None:
    calls: list[str] = []

    class FlakyProvider(FakeProvider):
        def generate(self, call):  # type: ignore[no-untyped-def]
            calls.append(call.model)
            if len(calls) == 1:
                from d0c_sight.llm.protocol import ProviderError

                msg = "503"
                raise ProviderError(msg)
            return super().generate(call)

    result, used = run_case(
        gold_case(),
        FlakyProvider(payload=payload_with()),
        _chunks(),
        LlmConfig(),
    )

    assert result.failed is False
    assert result.retries == 1
    assert used == 2


def test_run_result_is_incomplete_when_a_case_fails() -> None:
    """실패 문항이 하나라도 있으면 그 실행 전체가 불완전하다."""
    provider = FakeProvider(fail_models=frozenset({LlmConfig().model}))
    run = run_evalset(an_evalset(gold_case()), provider, _chunks(), max_retries=0)

    assert run.complete is False
    assert run.failed_case_ids == ("c1",)


def test_incomplete_run_reports_no_pass_rate() -> None:
    """27/30 만 보고 "87% 통과"라고 하면 틀린 숫자다. None 이 정답이다."""
    provider = FakeProvider(fail_models=frozenset({LlmConfig().model}))
    run = run_evalset(an_evalset(gold_case()), provider, _chunks(), max_retries=0)

    assert run.pass_rate is None
    assert run.to_dict()["pass_rate"] is None
    assert run.to_dict()["complete"] is False


def test_complete_run_reports_a_pass_rate() -> None:
    run = run_evalset(
        an_evalset(gold_case()),
        FakeProvider(payload=payload_with()),
        _chunks(),
    )
    assert run.complete is True
    assert run.pass_rate == 1.0


def test_run_records_what_produced_it() -> None:
    """모델·프롬프트 버전·설정이 없으면 결과를 비교할 수 없다."""
    run = run_evalset(
        an_evalset(gold_case()),
        FakeProvider(payload=payload_with()),
        _chunks(),
    )
    assert run.prompt_version == PROMPT_VERSION
    assert run.model == LlmConfig().model
    assert run.settings["thinking_budget"] == 0
    assert run.evalset_version == "v1"


def test_check_totals_expose_per_item_results() -> None:
    """총점만 보면 어느 항목이 뒤집혔는지 놓친다."""
    run = run_evalset(
        an_evalset(gold_case()),
        FakeProvider(payload=payload_with()),
        _chunks(),
    )
    totals = run.check_totals()
    assert totals[CheckId.GOLD_CITED.value] == (1, 1)
    assert set(totals) >= {CheckId.QUOTES_VERIFIED.value, CheckId.SCHEMA_VALID.value}


def test_cache_avoids_a_second_api_call(tmp_path: Path) -> None:
    """무료 티어 RPD 로는 캐시가 없으면 하루에 몇 번 못 돌린다."""
    cache = ResultCache(tmp_path)
    provider = FakeProvider(payload=payload_with())

    first, calls_1 = run_case(gold_case(), provider, _chunks(), LlmConfig(), cache)
    second, calls_2 = run_case(gold_case(), provider, _chunks(), LlmConfig(), cache)

    assert calls_1 == 1
    assert calls_2 == 0
    assert second.from_cache is True
    assert [c.check_id for c in second.checks] == [c.check_id for c in first.checks]


def test_cache_does_not_serve_a_different_prompt_version(tmp_path: Path) -> None:
    """캐시 키에 프롬프트 버전이 들어 있는지 실제 경로로 확인한다."""
    cache = ResultCache(tmp_path)
    case = gold_case()
    cache.put(cache_key(case, LlmConfig().model, "OTHER-VERSION", SETTINGS), {"bogus": True})

    _, calls = run_case(case, FakeProvider(payload=payload_with()), _chunks(), LlmConfig(), cache)

    assert calls == 1, "다른 프롬프트 버전의 캐시를 먹었다"
