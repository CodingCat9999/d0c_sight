"""공급사와 모델 설정.

공급사·모델을 코드가 아니라 설정으로 분리한다. Phase 9 모델 라우팅이 이 추상화를
그대로 쓴다. 지금 구현체는 gemini 하나뿐이며(ADR 0002 의 N=1 원칙), 두 번째 공급사가
생길 때 인터페이스를 고친다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

#: 기본 모델. 측정 없이 고른 잠정 선택이며 Phase 3 이후 재평가한다(ADR 0004).
DEFAULT_MODEL = "gemini-3.7-flash"

#: 가용성 폴백 후보. 품질 순서가 아니라 "살아 있는 모델" 순서다(ADR 0006).
DEFAULT_FALLBACKS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")

#: thinking 을 끄는 이유는 스키마의 reasoning 필드가 같은 일을 하기 때문이다.
#: 둘 다 켜면 같은 추론에 토큰이 이중으로 나가고, 무료 티어에서는 그것이 곧 쿼터다.
DEFAULT_THINKING_BUDGET = 0

DEFAULT_MAX_OUTPUT_TOKENS = 4096

#: 샘플링 온도. 0 으로 고정한다.
#:
#: 지정하지 않으면 공급사 기본값(비결정적)이 쓰인다. 실제로 같은 20문항을 같은 모델·
#: 같은 프롬프트로 두 번 돌렸더니 통과율이 90% 와 75% 로 갈렸다. 그 상태에서는 어떤
#: 변경이 개선인지 잡음인지 구별할 수 없다. 0 이어도 완전한 결정론은 아니지만 변동
#: 폭은 크게 줄어든다.
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """어느 공급사의 어느 모델을 어떻게 부를 것인가."""

    provider: str = "gemini"
    model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACKS
    thinking_budget: int = DEFAULT_THINKING_BUDGET
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    api_key_env: str = "GEMINI_API_KEY"

    @property
    def candidates(self) -> tuple[str, ...]:
        """시도 순서. 첫 모델이 죽으면 다음으로 넘어간다. 중복은 제거한다."""
        seen: dict[str, None] = {}
        for name in (self.model, *self.fallback_models):
            seen.setdefault(name, None)
        return tuple(seen)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LlmConfig:
        """환경변수로 설정을 덮어쓴다. 값이 없으면 기본값을 쓴다."""
        src = os.environ if env is None else env
        cfg = cls()
        model = src.get("D0C_LLM_MODEL")
        if model:
            cfg = replace(cfg, model=model)
        raw_fallbacks = src.get("D0C_LLM_FALLBACKS")
        if raw_fallbacks is not None:
            names = tuple(n.strip() for n in raw_fallbacks.split(",") if n.strip())
            cfg = replace(cfg, fallback_models=names)
        budget = src.get("D0C_LLM_THINKING_BUDGET")
        if budget:
            cfg = replace(cfg, thinking_budget=int(budget))
        return cfg


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """공급사에 보내는 한 번의 요청."""

    model: str
    system_instruction: str
    prompt: str
    response_schema: dict[str, object] = field(default_factory=dict)
    thinking_budget: int = DEFAULT_THINKING_BUDGET
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


@dataclass(frozen=True, slots=True)
class RawGeneration:
    """공급사가 돌려준 원문과 계측값. 아직 도메인 모델이 아니다."""

    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: int
