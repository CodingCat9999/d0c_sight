"""검색 지표 — recall@k 와 MRR.

**지금은 계산기만 만든다.** 검색이 없으므로(Phase 4) 기준선은 여기서 재지 않는다.

k 는 실제로 프롬프트에 넣을 청크 개수로 고정한다. 100개를 찾아도 10개만 쓰면
recall@100 은 의미가 없다. k 를 결과에 함께 기록해 나중에 값을 바꿔도 과거와 비교할
수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

#: 프롬프트에 넣을 청크 개수. Phase 4 에서 검색이 붙을 때 실측으로 재검토한다.
DEFAULT_K = 5


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """한 문항의 검색 성적. k 를 함께 들고 다닌다."""

    k: int
    recall_at_k: float
    reciprocal_rank: float
    gold_total: int
    gold_found: int


def score(retrieved: Sequence[str], gold: Collection[str], k: int = DEFAULT_K) -> RetrievalScore:
    """상위 k 개 안에서 정답을 몇 개 찾았고 첫 정답이 몇 위인가.

    정답이 없는 문항(no_answer)은 recall 을 1.0 으로 둔다 — 찾을 것이 없으므로
    "다 찾았다"가 맞고, 그렇게 하지 않으면 no_answer 문항이 평균을 끌어내린다.
    """
    gold_set = set(gold)
    top = list(retrieved[:k])
    if not gold_set:
        return RetrievalScore(k=k, recall_at_k=1.0, reciprocal_rank=1.0, gold_total=0, gold_found=0)

    found = sum(1 for cid in top if cid in gold_set)
    rr = 0.0
    for rank, cid in enumerate(top, start=1):
        if cid in gold_set:
            rr = 1.0 / rank
            break
    return RetrievalScore(
        k=k,
        recall_at_k=found / len(gold_set),
        reciprocal_rank=rr,
        gold_total=len(gold_set),
        gold_found=found,
    )


def mean_recall(scores: Sequence[RetrievalScore]) -> float:
    return sum(s.recall_at_k for s in scores) / len(scores) if scores else 0.0


def mean_reciprocal_rank(scores: Sequence[RetrievalScore]) -> float:
    return sum(s.reciprocal_rank for s in scores) / len(scores) if scores else 0.0
