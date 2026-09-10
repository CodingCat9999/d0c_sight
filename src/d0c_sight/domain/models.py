"""도메인 모델 — 모든 소스가 수렴하는 계약.

이 모듈은 프로젝트 내부의 어떤 것도 import 하지 않는다. 상위 계층이 소스를 몰라도
되게 하는 것이 이 계층의 존재 이유다.

모델은 실물을 파싱해본 뒤에 정의했다(ADR 0003). 실측 없이 정의했다면 설정
파라미터를 표로 모델링했을 것이고, 그것은 문서 1,148개 중 어디에도 맞지 않았을 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BlockKind(StrEnum):
    """콘텐츠 블록의 종류. 청킹의 원자 단위를 결정한다."""

    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    DEFINITION = "definition"


@dataclass(frozen=True, slots=True)
class DocVersion:
    """문서 버전.

    major 와 full 을 분리하는 이유는 둘의 쓰임이 다르기 때문이다. 질문은 major
    단위로 들어오고("PostgreSQL 18 에서…"), 재현성은 full 단위로 걸린다
    (18.6 과 18.7 의 문서는 다를 수 있다).
    """

    major: str
    full: str

    def __post_init__(self) -> None:
        if not self.full.startswith(f"{self.major}."):
            msg = f"full({self.full})이 major({self.major})로 시작하지 않는다"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """이 내용이 어디서 왔는가. 근거 추적의 최소 단위.

    Phase 6 의 근거 추적과 프로젝트 2 의 학습 데이터가 모두 이 구조 위에 선다.
    답변에서 이 참조를 따라가 원문의 해당 절에 도달할 수 있어야 한다.
    """

    source_id: str
    doc_id: str
    section_id: str
    version: DocVersion
    url: str


@dataclass(frozen=True, slots=True)
class Block:
    """섹션 안의 콘텐츠 블록 하나.

    subtype 은 원본의 의미 구분을 보존한다. `<pre>` 의 programlisting(예제 코드),
    synopsis(명령 문법), screen(실행 출력)은 검색에서 서로 다른 질문에 답한다.
    """

    kind: BlockKind
    text: str
    subtype: str | None = None
    heading: str | None = None


@dataclass(frozen=True, slots=True)
class Section:
    """앵커를 가진 문서의 한 절. 근거 링크의 단위다."""

    section_id: str
    title: str
    level: int
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class Document:
    """소스에서 가져와 구조화한 문서 하나."""

    doc_id: str
    title: str
    version: DocVersion
    url: str
    sections: tuple[Section, ...]


@dataclass(frozen=True, slots=True)
class ChunkLimits:
    """이 청크를 만들 때 적용된 임계값.

    임계값을 결과에 실어 보낸다(BRIEF §6.5). 나중에 값을 바꿔도 과거 결과와
    비교할 수 있어야 한다. 숫자만 남으면 근거 없이 바뀐다.
    """

    target_chars: int
    max_chars: int
    table_max_bytes: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """검색과 임베딩의 단위.

    part / of 는 원자 단위가 임계값을 넘어 쪼개진 경우를 드러낸다. 측정 한계를
    결과와 함께 내보내기 위한 것이다(BRIEF §6.6) — 사용자와 LLM 둘 다 이 청크가
    잘린 조각이라는 사실을 알아야 한다.
    """

    chunk_id: str
    text: str
    kind: BlockKind
    ref: SourceRef
    limits: ChunkLimits
    heading: str | None = None
    subtype: str | None = None
    part: int = 1
    of: int = 1

    def __post_init__(self) -> None:
        if self.part < 1 or self.of < 1 or self.part > self.of:
            msg = f"part/of 가 올바르지 않다: part={self.part}, of={self.of}"
            raise ValueError(msg)

    @property
    def is_split(self) -> bool:
        """원자 단위가 쪼개졌는가. 그렇다면 이 청크만으로는 불완전하다."""
        return self.of > 1
