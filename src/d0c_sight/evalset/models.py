"""평가셋 데이터 구조.

**점수가 아니라 판정이다.** 모든 채점 항목은 예/아니오다. "이 답변 4점" 같은 척도는
재현되지 않는다 — 두 사람이 독립적으로 매기면 다른 값이 나오고, 그러면 개선인지
잡음인지 구별할 수 없다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class CaseType(StrEnum):
    ERROR = "error"
    CONFIG = "config"
    CONCEPT = "concept"
    #: 문서에 답이 없어야 정상인 질문. 답을 지어내지 않는지 보는 경로다.
    NO_ANSWER = "no_answer"


class SearchAxis(StrEnum):
    """Phase 5 하이브리드 검색의 효과를 재기 위한 축."""

    KEYWORD_ONLY = "keyword_only"
    SEMANTIC_ONLY = "semantic_only"
    BOTH = "both"
    MULTI_CHUNK = "multi_chunk"


class Trap(StrEnum):
    SIMILAR_PARAMETER = "similar_parameter"
    EXCEPTION_CLAUSE = "exception_clause"
    VERSION_DEPENDENT = "version_dependent"


class Provenance(StrEnum):
    """이 문항이 어디서 왔는가.

    `DOCUMENTATION` 은 공식 문서에서 기계적으로 뽑은 것이다. 이 비중이 커지면
    문서를 문서로 검증하는 셈이 되어 평가셋이 쉬워진다. 절반을 넘기지 않는다.
    """

    GITHUB_ISSUE = "github_issue"
    MAILING_LIST = "mailing_list"
    DOCUMENTATION = "documentation"


class Split(StrEnum):
    """30문항 전부로 계속 튜닝하면 그 30개만 잘하는 시스템이 된다."""

    IMPROVE = "improve"
    HOLDOUT = "holdout"


def content_hash(text: str) -> str:
    """정답 청크 본문의 해시. 앞 16자만 쓴다 — 충돌 위험보다 가독성이 낫다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class GoldChunk:
    """정답 청크 하나.

    해시를 함께 저장하는 이유는 **id 는 살아 있는데 본문이 바뀌는** 경우가 실재하기
    때문이다. 18.4→18.6 실측에서 id 는 100% 유지됐지만 58개(0.35%)는 내용이 바뀌었다.
    존재 확인만으로는 잡히지 않고, 라벨이 조용히 틀려진다.
    """

    chunk_id: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class EvalCase:
    """평가 문항 하나."""

    case_id: str
    question: str
    version: str
    case_type: CaseType
    search_axis: SearchAxis
    gold_chunks: tuple[GoldChunk, ...]
    split: Split
    provenance: Provenance
    #: 왜 이 케이스인가. 없으면 나중에 무엇을 검증하려던 것인지 알 수 없다.
    rationale: str
    source_url: str | None = None
    traps: tuple[Trap, ...] = ()
    #: 이 문항에만 적용할 judge 항목. 기계 채점 항목은 자동으로 붙는다.
    judge_items: tuple[str, ...] = ()
    #: 정답 청크의 내용이 바뀌어 사람이 다시 봐야 하는 상태.
    needs_review: bool = False
    review_note: str = ""

    def __post_init__(self) -> None:
        if self.case_type is CaseType.NO_ANSWER and self.gold_chunks:
            msg = f"{self.case_id}: no_answer 문항에 정답 청크가 있다"
            raise ValueError(msg)
        if self.case_type is not CaseType.NO_ANSWER and not self.gold_chunks:
            msg = f"{self.case_id}: 정답 청크가 없다"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EvalSet:
    """평가셋 전체.

    버전을 붙이는 이유는 문서가 개정되거나 청킹 전략이 바뀌면 정답 청크가 무효가 되기
    때문이다. 그때 무엇이 깨졌는지 알아야 한다.
    """

    version: str
    created: str
    corpus_version: str
    cases: tuple[EvalCase, ...] = field(default_factory=tuple)

    def by_split(self, split: Split) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if c.split is split)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(c.case_id for c in self.cases)

    def __post_init__(self) -> None:
        seen = [c.case_id for c in self.cases]
        dupes = {i for i in seen if seen.count(i) > 1}
        if dupes:
            msg = f"case_id 가 중복됐다: {sorted(dupes)}"
            raise ValueError(msg)
