"""계층 가드의 짝 테스트.

두 방향을 모두 확인한다(BRIEF §6.4).
  - 위반을 심으면 가드가 잡는가
  - 정당한 import 를 가드가 과잉 차단하지 않는가
"""

from __future__ import annotations

from pathlib import Path

import d0c_sight
from tests.guards.layering import (
    ALLOWED_IMPORTS,
    count_modules,
    discovered_layers,
    find_violations,
)

PACKAGE_ROOT = Path(d0c_sight.__file__).parent


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ─── 실제 트리 ────────────────────────────────────────────────


def test_package_root_has_modules_to_check() -> None:
    """부재 단언 앞의 존재 단언. 트리를 못 찾으면 아래 테스트는 공허하게 통과한다."""
    assert count_modules(PACKAGE_ROOT) > 0
    assert discovered_layers(PACKAGE_ROOT)


def test_every_layer_is_registered() -> None:
    """ALLOWED_IMPORTS 에 없는 계층이 생기면 검사에서 조용히 빠진다."""
    unregistered = discovered_layers(PACKAGE_ROOT) - set(ALLOWED_IMPORTS)
    assert not unregistered, f"미등록 계층: {sorted(unregistered)}"


def test_real_tree_has_no_violations() -> None:
    violations = find_violations(PACKAGE_ROOT)
    assert not violations, "\n".join(str(v) for v in violations)


# ─── 위반을 심으면 잡는가 ──────────────────────────────────────


def test_catches_absolute_import_from_domain(tmp_path: Path) -> None:
    _write(tmp_path, "domain/models.py", "from d0c_sight.sources import protocol\n")

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].layer == "domain"
    assert violations[0].imported_layer == "sources"
    assert violations[0].lineno == 1


def test_catches_relative_import_escaping_the_layer(tmp_path: Path) -> None:
    """`from ...pipeline import x` 같은 우회도 잡아야 한다."""
    _write(tmp_path, "domain/sub/models.py", "from ...pipeline import chunk\n")

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].imported_layer == "pipeline"


def test_catches_plain_import_statement(tmp_path: Path) -> None:
    _write(tmp_path, "sources/adapter.py", "import d0c_sight.pipeline.chunk\n")

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].imported_layer == "pipeline"


def test_catches_violation_in_package_init(tmp_path: Path) -> None:
    _write(tmp_path, "domain/__init__.py", "from d0c_sight.cli import main\n")

    violations = find_violations(tmp_path)

    assert len(violations) == 1


# ─── 과잉 차단하지 않는가 ──────────────────────────────────────


def test_allows_downward_dependency(tmp_path: Path) -> None:
    _write(tmp_path, "pipeline/chunk.py", "from d0c_sight.domain import models\n")

    assert count_modules(tmp_path) == 1  # 파일을 실제로 읽었는지 먼저 확인
    assert find_violations(tmp_path) == []


def test_allows_import_within_the_same_layer(tmp_path: Path) -> None:
    _write(tmp_path, "sources/postgres_docs/fetch.py", "from ..protocol import Source\n")

    assert count_modules(tmp_path) == 1
    assert find_violations(tmp_path) == []


def test_allows_third_party_and_stdlib_imports(tmp_path: Path) -> None:
    _write(tmp_path, "domain/models.py", "import ast\nfrom lxml import etree\n")

    assert count_modules(tmp_path) == 1
    assert find_violations(tmp_path) == []
