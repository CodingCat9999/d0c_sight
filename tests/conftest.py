"""테스트 0개는 실패다 — 가드.

glob 오타나 경로 변경으로 테스트가 통째로 사라져도 CI 가 초록으로 남는 상황을
막는다(BRIEF §6.3). pytest 는 미수집 시 exit code 5 를 내지만, 호출 측이 5 를
실패로 다루지 않으면 조용히 통과한다. CI 스텝과 이 훅 두 층으로 막는다.

`trylast=True` 가 중요하다. 이 훅은 기본 순서에서 pytest 내장 `-k` / `-m`
deselection 보다 **먼저** 호출되어, 필터 전 items 를 보게 된다. 그러면
플러그인이 전부 deselect 한 경우를 놓치고, 아래 필터 예외 분기도 도달할 수 없는
죽은 코드가 된다. 실제로 변이 검증에서 그렇게 드러나 순서를 고정했다.

`-k` / `-m` 으로 일부러 좁힌 경우와 `--collect-only` 는 제외한다. 과잉 차단하면
우회하게 되고, 우회된 가드는 없는 가드다(BRIEF §6.5).
"""

from __future__ import annotations

import pytest

NO_TESTS_MESSAGE = "테스트가 하나도 수집되지 않았다"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if items:
        return
    if config.option.keyword or config.option.markexpr or config.option.collectonly:
        return
    raise pytest.UsageError(
        f"{NO_TESTS_MESSAGE}. testpaths 설정과 파일명 규칙(test_*.py)을 확인하라. "
        "0개 통과는 통과가 아니라 미실행이다."
    )
