"""계층 의존 방향 가드.

    domain  ←  sources  ←  pipeline  ←  cli

`domain` 은 프로젝트 내부의 어떤 모듈도 import 하지 않는다. 나머지 계층은
자기보다 아래(왼쪽)만 import 한다. 이 규칙이 깨지면 순환 의존이 생기고
상위 계층이 소스 구현을 알게 된다.

검사 로직을 순수 함수로 분리한 이유는, 짝 테스트에서 일부러 위반하는 트리를
만들어 "이 가드가 실제로 위반을 잡는가"를 검증하기 위해서다(BRIEF §6.4).
가드만 있고 그것을 깨뜨려본 적이 없으면 그 가드는 없는 것과 같다.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "d0c_sight"

#: 각 계층이 import 해도 되는 내부 계층. domain 이 비어 있는 것이 이 표의 핵심이다.
ALLOWED_IMPORTS: dict[str, frozenset[str]] = {
    "domain": frozenset(),
    "sources": frozenset({"domain"}),
    "pipeline": frozenset({"domain", "sources"}),
    # llm 은 청크가 어디서 왔는지 알 필요가 없다. sources 도 pipeline 도 import 하지 않는다.
    "llm": frozenset({"domain"}),
    "cli": frozenset({"domain", "sources", "pipeline", "llm"}),
}


@dataclass(frozen=True)
class Violation:
    """계층 규칙 위반 한 건."""

    module: str
    layer: str
    imported_layer: str
    lineno: int

    def __str__(self) -> str:
        return (
            f"{self.module}:{self.lineno} — "
            f"'{self.layer}' 계층이 '{self.imported_layer}' 을(를) import 한다"
        )


def _split(path: Path, package_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(모듈 경로 조각, 그 모듈이 속한 패키지 경로 조각)."""
    rel = path.relative_to(package_root)
    if rel.name == "__init__.py":
        parts = rel.parts[:-1]
        return parts, parts
    parts = (*rel.parts[:-1], rel.stem)
    return parts, parts[:-1]


def _absolute_target(name: str) -> str | None:
    """'d0c_sight.sources.x' → 'sources'. 외부 패키지면 None."""
    head, _, rest = name.partition(".")
    if head != PACKAGE or not rest:
        return None
    return rest.partition(".")[0]


def _relative_target(pkg: tuple[str, ...], level: int, module: str | None) -> str | None:
    """상대 import 를 계층 이름으로 해석한다. `from ...domain import x` 같은 우회를 막는다."""
    keep = len(pkg) - (level - 1)
    base = pkg[:keep] if keep > 0 else ()
    target = (*base, *module.split(".")) if module else base
    return target[0] if target else None


def _import_targets(tree: ast.Module, pkg: tuple[str, ...]) -> Iterator[tuple[str | None, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield _absolute_target(alias.name), node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield _relative_target(pkg, node.level, node.module), node.lineno
            elif node.module:
                yield _absolute_target(node.module), node.lineno


def discovered_layers(package_root: Path) -> set[str]:
    """트리에서 실제로 발견된 계층 이름.

    ALLOWED_IMPORTS 에 없는 계층이 생기면 검사에서 조용히 빠지므로,
    짝 테스트가 이 집합을 함께 단언한다.
    """
    layers: set[str] = set()
    for path in package_root.rglob("*.py"):
        parts, _ = _split(path, package_root)
        if parts:
            layers.add(parts[0])
    return layers


def find_violations(package_root: Path) -> list[Violation]:
    """계층 규칙 위반 전체. 위반이 없으면 빈 리스트."""
    violations: list[Violation] = []
    for path in sorted(package_root.rglob("*.py")):
        parts, pkg = _split(path, package_root)
        if not parts:
            continue
        layer = parts[0]
        allowed = ALLOWED_IMPORTS.get(layer)
        if allowed is None:
            continue  # 미등록 계층은 discovered_layers() 쪽에서 잡는다
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target, lineno in _import_targets(tree, pkg):
            if target is None or target == layer or target in allowed:
                continue
            violations.append(
                Violation(
                    module=".".join((PACKAGE, *parts)),
                    layer=layer,
                    imported_layer=target,
                    lineno=lineno,
                )
            )
    return violations


def count_modules(package_root: Path) -> int:
    """검사 대상 모듈 수. 부재 단언 앞의 존재 단언에 쓴다(BRIEF §6.4)."""
    return sum(1 for _ in package_root.rglob("*.py"))
