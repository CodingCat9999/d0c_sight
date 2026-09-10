"""Source 프로토콜 — 외부 문서를 도메인 모델로 옮기는 계약.

**이 인터페이스는 아직 검증되지 않았다.** 구현체가 postgres_docs 하나뿐이라
이것이 옳은 모양인지 알 방법이 없다. 두 번째 소스(메일링 리스트)를 넣을 때 고치는
것을 전제로 두었다(ADR 0002).

따라서 이 프로토콜이 맞다고 가정하고 그 위에 여러 계층을 쌓지 않는다. 지금 이것이
하는 일은 상위 계층이 PostgreSQL 문서 구조를 직접 알지 못하게 막는 것뿐이다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from d0c_sight.domain.models import Document, DocVersion


@runtime_checkable
class Source(Protocol):
    """문서 소스 어댑터."""

    @property
    def source_id(self) -> str:
        """소스 식별자. SourceRef 에 기록되어 근거 추적에 쓰인다."""
        ...

    @property
    def version(self) -> DocVersion:
        """이 소스가 다루는 문서 버전."""
        ...

    def fetch(self, dest_dir: Path) -> Path:
        """원본을 가져와 dest_dir 아래에 두고 그 경로를 돌려준다.

        이미 받아둔 것이 있으면 다시 받지 않는다. 수집은 재현 가능해야 한다.
        """
        ...

    def parse(self, raw: Path) -> Iterator[Document]:
        """가져온 원본을 Document 로 변환한다.

        인덱싱 대상이 아닌 문서는 여기서 걸러 내보내지 않는다. 필터를 함수
        시그니처 밖에 두지 않기 위해서다 — 상위 계층이 각자 거르면 한 군데를
        빠뜨렸을 때 조용히 틀린다(BRIEF §6.2).
        """
        ...
