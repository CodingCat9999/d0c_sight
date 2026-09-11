"""비밀값과 원문이 새지 않는가 — 가드의 짝 테스트.

키가 로그에 한 번 새면 되돌릴 수 없다. 새지 않았다는 것은 테스트로만 확인할 수 있다.
프롬프트 원문도 마찬가지다 — 무료 티어에 보낸 내용이 로그 파일에 그대로 쌓이면
그것 자체가 별도의 노출 경로가 된다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import pytest

from d0c_sight.llm.config import LlmConfig
from d0c_sight.llm.diagnose import diagnose
from d0c_sight.llm.gemini import GeminiProvider, scrub
from d0c_sight.llm.protocol import ProviderError
from tests.unit.test_llm import FakeProvider, make_chunk, payload_with

FAKE_KEY = "AQ.FakeKeyForTesting_0123456789abcdef"
SECRET_MARKER = "SUPERSECRET_PROMPT_BODY_MARKER"


class ExplodingClient:
    """예외 메시지에 키를 실어 보내는 최악의 SDK 를 흉내낸다."""

    class _Models:
        def generate_content(self, **_: object) -> object:
            msg = f"401 Unauthorized: key={FAKE_KEY} rejected"
            raise RuntimeError(msg)

    models = _Models()


# ─── 마스킹 자체 ──────────────────────────────────────────────


def test_scrub_removes_the_secret() -> None:
    assert FAKE_KEY not in scrub(f"boom key={FAKE_KEY}", FAKE_KEY)


def test_scrub_keeps_the_surrounding_message() -> None:
    """전부 지워버리면 진단이 불가능해진다. 키만 지운다."""
    out = scrub(f"401 Unauthorized: key={FAKE_KEY} rejected", FAKE_KEY)
    assert "401 Unauthorized" in out
    assert "rejected" in out


def test_scrub_is_a_noop_without_a_secret() -> None:
    assert scrub("plain message", None) == "plain message"
    assert scrub("plain message", "") == "plain message"


# ─── 실제 경로 ────────────────────────────────────────────────


def test_provider_error_does_not_carry_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK 예외를 그대로 흘리면 키가 스택 트레이스에 남는다."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    provider = GeminiProvider(client=ExplodingClient())

    from d0c_sight.llm.config import ProviderCall

    with pytest.raises(ProviderError) as exc:
        provider.generate(ProviderCall(model="m", system_instruction="s", prompt="p"))

    text = f"{exc.value}"
    assert "401 Unauthorized" in text, "원인을 알 수 없게 만들면 안 된다"
    assert FAKE_KEY not in text
    assert exc.value.__cause__ is None, "원본 예외가 체인으로 남으면 키도 함께 남는다"


def test_logs_do_not_contain_the_prompt_body(caplog: pytest.LogCaptureFixture) -> None:
    """관측은 하되 원문은 남기지 않는다."""
    provider = FakeProvider(payload=payload_with())
    with caplog.at_level(logging.INFO, logger="d0c_sight.llm"):
        diagnose(f"error {SECRET_MARKER}", [make_chunk()], provider, LlmConfig())

    assert caplog.records, "로그가 아예 없으면 이 단언은 공허하다"
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "stop=" in blob, "관측값은 남아야 한다"
    assert SECRET_MARKER not in blob


def test_serialised_result_does_not_contain_the_prompt_body() -> None:
    """저장물에도 원문이 통째로 들어가지 않는다."""
    provider = FakeProvider(payload=payload_with())
    result = diagnose(f"error {SECRET_MARKER}", [make_chunk()], provider, LlmConfig())

    blob = json.dumps(asdict(result), ensure_ascii=False)
    assert '"model"' in blob, "결과가 비어 있으면 이 단언은 공허하다"
    assert SECRET_MARKER not in blob


def test_prompt_was_actually_built_with_the_marker() -> None:
    """부재 단언 앞의 존재 단언 — 마커가 프롬프트에 실제로 들어갔는지 먼저 확인한다.

    이것이 없으면 위 두 테스트는 '마커가 어디에도 없다'는 이유로 통과할 수 있다.
    """
    provider = FakeProvider(payload=payload_with())
    diagnose(f"error {SECRET_MARKER}", [make_chunk()], provider, LlmConfig())
    assert provider.calls
    assert SECRET_MARKER in provider.calls[0].prompt
