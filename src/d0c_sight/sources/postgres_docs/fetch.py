"""PostgreSQL 공식 문서 tarball 수집.

공식 docs 페이지(postgresql.org/docs/)가 링크하는 다운로드는 PDF 뿐이다. 사전 빌드된
HTML tarball 은 릴리스 디렉터리에만 있다(ADR 0001).

**해시 확인은 무결성 검증이 아니라 재현성 고정이다.** 릴리스 디렉터리의 `.sha256` 는
소스 tarball 에만 붙어 있고 docs tarball 에는 없다. 여기서 하는 일은 "우리가 처음 받은
것과 같은 파일인가"를 확인하는 것이며, "이것이 진짜 PostgreSQL 배포본인가"는 보장하지
않는다. 후자는 전송 경로가 HTTPS 라는 것에 의존한다.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

from d0c_sight.domain.models import DocVersion

VERSION = DocVersion(major="18", full="18.6")

DOCS_URL = "https://ftp.postgresql.org/pub/source/v18.6/postgresql-18.6-docs.tar.gz"
TARBALL_NAME = "postgresql-18.6-docs.tar.gz"

#: 2026-09-10 에 받은 파일의 해시. 위 docstring 대로 재현성 고정용이다.
EXPECTED_SHA256 = "0419dec0d3b7ca55a80c0519a1dd88a8d172019ef9841288889f0281ea1f97ed"

#: tarball 안에서 HTML 이 들어 있는 경로.
HTML_SUBPATH = Path("postgresql-18.6/doc/src/sgml/html")

_READ_CHUNK = 1 << 20


class FetchError(RuntimeError):
    """수집 실패. 해시 불일치와 예상 구조 부재를 모두 포함한다."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str = EXPECTED_SHA256) -> None:
    """받은 파일이 기록된 것과 같은지 확인한다.

    다르면 예외를 던진다. 조용히 진행하면 인덱스가 어느 문서에서 왔는지 알 수 없게
    되고, Phase 3 이후의 평가 결과가 무엇을 잰 것인지도 알 수 없게 된다.
    """
    actual = sha256_of(path)
    if actual != expected:
        msg = (
            f"tarball 해시가 기록된 값과 다르다.\n"
            f"  기대: {expected}\n"
            f"  실제: {actual}\n"
            f"새 마이너 릴리스로 파일이 바뀌었다면 EXPECTED_SHA256 을 갱신하고 "
            f"그 사실을 커밋에 남긴다. 인덱스가 어느 문서에서 왔는지 추적할 수 "
            f"있어야 한다."
        )
        raise FetchError(msg)


def download(dest_dir: Path, url: str = DOCS_URL) -> Path:
    """tarball 을 받는다. 이미 있고 해시가 맞으면 다시 받지 않는다."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / TARBALL_NAME
    if target.exists():
        verify_sha256(target)
        return target
    urllib.request.urlretrieve(url, target)
    verify_sha256(target)
    return target


def extract(tarball: Path, dest_dir: Path) -> Path:
    """tarball 을 풀고 HTML 디렉터리 경로를 돌려준다."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        # filter="data" 는 절대경로·상위경로 탈출 항목을 거부한다.
        # Python 3.14 부터 기본값이지만 3.12 에서는 명시해야 한다.
        tar.extractall(dest_dir, filter="data")
    html_dir = dest_dir / HTML_SUBPATH
    if not html_dir.is_dir():
        msg = (
            f"tarball 안에 예상 경로가 없다: {HTML_SUBPATH}\n"
            f"릴리스 구조가 바뀌었을 수 있다. ADR 0001 의 재검토 조건에 해당한다."
        )
        raise FetchError(msg)
    return html_dir


def fetch(dest_dir: Path) -> Path:
    """수집 전체. HTML 디렉터리 경로를 돌려준다."""
    tarball = download(dest_dir)
    return extract(tarball, dest_dir)
