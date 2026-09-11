"""Gemini 공급사 구현 — 현재 유일한 구현체.

무료 티어를 쓴다. **입출력이 모델 개선에 사용될 수 있으므로 공개 문서만 넣는다.**
PostgreSQL 공식 문서는 공개 자료다. 비공개 데이터를 이 경로로 보내지 않는다.

재시도는 SDK 가 한다. 실측한 기본값은 5회 시도 / 1초에서 시작하는 지수 백오프 /
최대 60초 / 지터 포함이고, 대상 상태코드는 408, 429, 500, 502, 503, 504 다.
429 가 포함되어 있어 무료 티어의 분당 제한도 SDK 가 처리한다. 그 위에 중복
구현하지 않는다 — 모델 폴백만 상위(diagnose)에서 얹는다.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from d0c_sight.llm.protocol import ProviderError

if TYPE_CHECKING:
    from d0c_sight.llm.config import ProviderCall, RawGeneration

PROVIDER_ID = "gemini"

_REDACTED = "***REDACTED***"


def scrub(text: str, secret: str | None) -> str:
    """로그와 예외에서 비밀값을 지운다.

    SDK 가 키를 예외에 넣지 않더라도 여기서 한 번 더 막는다. 키가 로그에 새면
    되돌릴 수 없고, 새지 않았다는 것은 테스트로만 확인할 수 있다.
    """
    if secret and secret in text:
        return text.replace(secret, _REDACTED)
    return text


class GeminiProvider:
    """google-genai SDK 어댑터."""

    def __init__(self, api_key_env: str = "GEMINI_API_KEY", client: Any = None) -> None:
        self._api_key = os.environ.get(api_key_env, "")
        if client is not None:
            self._client = client
            return
        if not self._api_key:
            msg = f"{api_key_env} 가 설정되지 않았다. .env 를 확인하라."
            raise ProviderError(msg)
        from google import genai

        self._client = genai.Client(api_key=self._api_key)

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    def generate(self, call: ProviderCall) -> RawGeneration:
        from google.genai import types

        from d0c_sight.llm.config import RawGeneration

        config = types.GenerateContentConfig(
            system_instruction=call.system_instruction,
            response_mime_type="application/json",
            response_schema=call.response_schema or None,
            thinking_config=types.ThinkingConfig(thinking_budget=call.thinking_budget),
            max_output_tokens=call.max_output_tokens,
            temperature=call.temperature,
            # 도구를 쓰지 않는다. 끄지 않으면 SDK 가 매 호출마다 AFC 권고 경고를 낸다.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        started = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=call.model, contents=call.prompt, config=config
            )
        except Exception as exc:  # SDK 예외를 그대로 흘리지 않는다 — 키가 섞일 수 있다
            detail = scrub(f"{type(exc).__name__}: {exc}", self._api_key)
            msg = f"{call.model} 호출 실패 — {detail}"
            raise ProviderError(msg) from None
        elapsed_ms = int((time.monotonic() - started) * 1000)

        usage = response.usage_metadata
        candidate = response.candidates[0] if response.candidates else None
        return RawGeneration(
            text=response.text or "",
            stop_reason=str(getattr(candidate, "finish_reason", "") or "unknown"),
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            latency_ms=elapsed_ms,
        )
