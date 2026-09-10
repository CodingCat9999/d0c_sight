"""'테스트 0개는 실패' 가드의 짝 테스트.

가드를 서브프로세스 pytest 로 실제 발동시켜 확인한다. 양방향으로 본다.
  - 수집이 0개면 정말 실패하는가
  - 정상적인 실행과 의도적인 필터를 과잉 차단하지 않는가
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from tests.conftest import NO_TESTS_MESSAGE

CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"

PASSING_TEST = "def test_ok() -> None:\n    assert True\n"


def _run_pytest(tmp: Path, *args: str) -> subprocess.CompletedProcess[str]:
    shutil.copy(CONFTEST, tmp / "conftest.py")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp), "-p", "no:cacheprovider", *args],
        capture_output=True,
        text=True,
        cwd=tmp,
        check=False,
    )


def test_conftest_guard_source_exists() -> None:
    """부재 단언 앞의 존재 단언. conftest 를 못 찾으면 아래가 전부 공허해진다."""
    assert CONFTEST.is_file()
    assert "pytest_collection_modifyitems" in CONFTEST.read_text(encoding="utf-8")


def test_zero_collected_tests_fails(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path)

    assert result.returncode != 0
    assert NO_TESTS_MESSAGE in result.stdout + result.stderr


def test_a_normal_run_is_not_blocked(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(PASSING_TEST, encoding="utf-8")

    result = _run_pytest(tmp_path)

    assert result.returncode == 0
    assert NO_TESTS_MESSAGE not in result.stdout + result.stderr


def test_deliberate_keyword_filter_is_not_blocked(tmp_path: Path) -> None:
    """`-k` 로 아무것도 안 골라도 가드는 발동하지 않아야 한다."""
    (tmp_path / "test_sample.py").write_text(PASSING_TEST, encoding="utf-8")

    result = _run_pytest(tmp_path, "-k", "no_such_test_name")

    assert NO_TESTS_MESSAGE not in result.stdout + result.stderr


def test_collect_only_is_not_blocked(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path, "--collect-only")

    assert NO_TESTS_MESSAGE not in result.stdout + result.stderr


def test_guard_sees_items_after_other_plugins_filtered(tmp_path: Path) -> None:
    """다른 플러그인이 전부 deselect 한 경우도 잡아야 한다.

    `trylast=True` 가 없으면 이 가드는 필터 전 items 를 보게 되어 여기를 놓친다.
    이 테스트가 그 순서를 고정한다.
    """
    (tmp_path / "test_sample.py").write_text(PASSING_TEST, encoding="utf-8")
    (tmp_path / "eater.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.clear()\n", encoding="utf-8"
    )

    result = _run_pytest(tmp_path, "-p", "eater")

    assert NO_TESTS_MESSAGE in result.stdout + result.stderr
