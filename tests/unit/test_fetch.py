"""수집 어댑터 테스트.

네트워크를 타지 않는다. 해시 확인과 tarball 추출은 순수하게 분리되어 있어 실제
다운로드 없이 검증할 수 있다. 네트워크 왕복 자체는 urlretrieve 한 줄이며 테스트하지
않는다 — 그 사실을 여기 적어 둔다. 테스트하지 않은 것을 테스트한 것처럼 두지 않는다.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from d0c_sight.sources.postgres_docs.fetch import (
    HTML_SUBPATH,
    FetchError,
    extract,
    sha256_of,
    verify_sha256,
)


def _make_tarball(tmp: Path, inner: Path, files: dict[str, str]) -> Path:
    """inner 경로 아래에 files 를 담은 tar.gz 를 만든다."""
    stage = tmp / "stage"
    for name, body in files.items():
        p = stage / inner / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    tarball = tmp / "docs.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(stage, arcname=".")
    return tarball


# ─── 해시 ─────────────────────────────────────────────────────


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"postgres" * 1000)
    assert sha256_of(f) == hashlib.sha256(b"postgres" * 1000).hexdigest()


def test_sha256_reads_files_larger_than_one_block(tmp_path: Path) -> None:
    """청크 단위로 읽으므로 1MB 경계를 넘는 파일을 명시적으로 본다."""
    body = b"x" * ((1 << 20) + 7)
    f = tmp_path / "big.bin"
    f.write_bytes(body)
    assert sha256_of(f) == hashlib.sha256(body).hexdigest()


def test_verify_passes_on_matching_hash(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    verify_sha256(f, hashlib.sha256(b"hello").hexdigest())  # 예외가 없어야 한다


def test_verify_rejects_mismatched_hash(tmp_path: Path) -> None:
    """가드가 실제로 막는지 확인한다. 통과 경로만 테스트하면 검증이 없는 것과 같다."""
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")

    with pytest.raises(FetchError, match="해시가 기록된 값과 다르다"):
        verify_sha256(f, "0" * 64)


def test_mismatch_message_shows_both_hashes(tmp_path: Path) -> None:
    """무엇이 어긋났는지 보이지 않으면 사용자는 원인을 찾을 수 없다."""
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    expected = "0" * 64

    with pytest.raises(FetchError) as exc:
        verify_sha256(f, expected)

    assert expected in str(exc.value)
    assert sha256_of(f) in str(exc.value)


# ─── 추출 ─────────────────────────────────────────────────────


def test_extract_returns_html_directory(tmp_path: Path) -> None:
    tarball = _make_tarball(tmp_path, HTML_SUBPATH, {"index.html": "<html/>"})

    html_dir = extract(tarball, tmp_path / "out")

    assert html_dir.is_dir()
    assert (html_dir / "index.html").read_text(encoding="utf-8") == "<html/>"


def test_extract_rejects_tarball_without_expected_layout(tmp_path: Path) -> None:
    """릴리스 구조가 바뀌면 조용히 빈 인덱스를 만들지 말고 실패해야 한다."""
    tarball = _make_tarball(tmp_path, Path("somewhere/else"), {"index.html": "<html/>"})

    with pytest.raises(FetchError, match="예상 경로가 없다"):
        extract(tarball, tmp_path / "out")
