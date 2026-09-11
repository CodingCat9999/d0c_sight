"""공급사 프로토콜.

**검증되지 않은 인터페이스다.** 구현체가 gemini 하나뿐이라 이 모양이 옳은지 알 수
없다. 두 번째 공급사를 붙일 때 고치는 것을 전제로 둔다(ADR 0002).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from d0c_sight.llm.config import ProviderCall, RawGeneration


class ProviderError(RuntimeError):
    """공급사 호출 실패. 메시지에 API 키가 들어가지 않아야 한다."""


@runtime_checkable
class LlmProvider(Protocol):
    """구조화된 JSON 을 돌려주는 생성 공급사."""

    @property
    def provider_id(self) -> str:
        """결과에 기록되는 공급사 식별자."""
        ...

    def generate(self, call: ProviderCall) -> RawGeneration:
        """한 번 호출한다. 재시도는 SDK 가 하고, 모델 폴백은 상위가 한다."""
        ...
