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
    """소스에서 가져와 구조화한 문서 하나.

    source_id 와 url 을 문서가 들고 있는 이유는, 상위 계층이 어느 소스에서 왔는지
    묻지 않고도 근거를 기록할 수 있게 하기 위해서다. 이것이 없으면 청킹 계층이
    구현체를 직접 알아야 한다.
    """

    source_id: str
    doc_id: str
    title: str
    version: DocVersion
    url: str
    sections: tuple[Section, ...]

    def ref_for(self, section: Section) -> SourceRef:
        """섹션 하나에 대한 근거 참조. URL 앵커는 여기서 한 번만 조립된다."""
        return SourceRef(
            source_id=self.source_id,
            doc_id=self.doc_id,
            section_id=section.section_id,
            version=self.version,
            url=f"{self.url}#{section.section_id}",
        )


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

    @property
    def quotable(self) -> str:
        """LLM 에게 보여주는 본문이자 인용 검증의 대상.

        렌더링과 검증이 **같은 문자열**을 써야 한다. 다르면 모델이 실제로 본 것을
        정직하게 인용해도 검증에서 걸린다. 실제로 그런 일이 있었다 — 컨텍스트에는
        heading 과 text 를 이어 보여주면서 검증은 text 에만 해서, 모델이
        "max_connections (integer)\nDetermines the…" 를 인용하자 위반으로 잡혔다.
        """
        return f"{self.heading}\n{self.text}" if self.heading else self.text


class Confidence(StrEnum):
    """원인 후보에 대한 모델의 확신도.

    대소문자만 다른 값을 만들지 않는다. 비교할 때도 대소문자를 무시한다 —
    스키마에 enum 을 걸어도 지정하지 않은 표기가 돌아올 수 있다.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def parse(cls, raw: str) -> Confidence:
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.LOW


class DropReason(StrEnum):
    """근거가 검증에서 걸러진 이유."""

    UNKNOWN_CHUNK_ID = "unknown_chunk_id"
    NOT_IN_CONTEXT = "not_in_context"
    QUOTE_NOT_FOUND = "quote_not_found"


@dataclass(frozen=True, slots=True)
class Evidence:
    """주장 하나를 뒷받침하는 근거.

    quote 를 함께 받는 이유는 Phase 6 의 문장 단위 근거 추적과 프로젝트 2 의 학습
    데이터가 (주장, 근거 문장) 쌍을 전제하기 때문이다. chunk_id 만으로는 모델이
    인용을 변형했는지 창작했는지 검증할 수 없다.
    """

    chunk_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class DroppedEvidence:
    """검증에서 걸러진 근거. 조용히 버리지 않고 결과에 실어 보낸다(BRIEF §6.6)."""

    chunk_id: str
    quote: str
    reason: DropReason


@dataclass(frozen=True, slots=True)
class Cause:
    """원인 후보 하나.

    근거를 답변 전체가 아니라 후보마다 붙인다. 어느 주장이 어느 문서에서 왔는지
    구별할 수 없으면 근거 추적이 성립하지 않는다.
    """

    description: str
    evidence: tuple[Evidence, ...]
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """구조화된 진단 결과."""

    reasoning: str
    causes: tuple[Cause, ...]
    checks: tuple[str, ...]
    version_basis: str
    insufficient_evidence: bool


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """어떤 공급사·모델이 어떻게 답했는가.

    공급사와 모델을 결과에 기록해야 Phase 9 라우팅에서 과거 결과를 비교할 수 있다.
    fallback 이 몇 번 일어났는지도 남긴다 — 가용성 문제가 조용히 묻히지 않게.
    """

    provider: str
    model: str
    attempted_models: tuple[str, ...]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: int

    @property
    def fallback_count(self) -> int:
        """첫 후보가 실패해 다음으로 넘어간 횟수."""
        return max(0, len(self.attempted_models) - 1)


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    """진단과 그 한계.

    dropped_evidence 가 비어 있지 않다면 모델이 검증되지 않는 근거를 댔다는 뜻이다.
    수치만 주고 한계를 숨기면 사용자는 그것을 확정 사실로 읽는다(BRIEF §6.6).
    """

    diagnosis: Diagnosis
    record: GenerationRecord
    dropped_evidence: tuple[DroppedEvidence, ...]
    model_declared_insufficient: bool = False

    @property
    def has_verified_evidence(self) -> bool:
        return any(c.evidence for c in self.diagnosis.causes)

    @property
    def insufficient_cause(self) -> str | None:
        """근거가 부족하다면 그것이 누구의 판단인가.

        "모델이 답하기를 거부했다"와 "우리 검증이 근거를 전부 걸렀다"는 전혀 다른
        실패다. 전자는 프롬프트나 컨텍스트 문제이고 후자는 환각이다. 구분되지 않으면
        무엇을 고쳐야 할지 알 수 없다.
        """
        if not self.diagnosis.insufficient_evidence:
            return None
        if self.model_declared_insufficient:
            return "model_declined"
        return "verification_removed_all" if self.dropped_evidence else "no_causes_returned"
