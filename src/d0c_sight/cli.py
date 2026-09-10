"""명령줄 진입점.

Phase 1 에서는 수집과 청킹만 한다. 검색도 LLM 도 없다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from d0c_sight.pipeline.run import chunk_documents, write_jsonl
from d0c_sight.sources.postgres_docs import fetch, parse

DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/processed/chunks.jsonl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="d0c-sight", description="근거 추적 가능한 인프라 에러 진단")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="문서를 수집하고 청크로 만든다")
    ing.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW, help="tarball 을 둘 위치")
    ing.add_argument("--out", type=Path, default=DEFAULT_OUT, help="청크 JSONL 출력 경로")
    ing.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="이미 풀어둔 HTML 디렉터리. 주면 다운로드를 건너뛴다",
    )
    return p


def _ingest(args: argparse.Namespace) -> int:
    html_dir: Path = args.html_dir or fetch.fetch(args.raw_dir)
    if not html_dir.is_dir():
        print(f"HTML 디렉터리가 없다: {html_dir}", file=sys.stderr)
        return 1

    docs = parse.parse_all(html_dir, fetch.VERSION)
    report = write_jsonl(chunk_documents(docs), args.out)

    print(f"출력 {args.out}")
    for line in report.as_lines():
        print(f"  {line}")
    if report.chunks == 0:
        print("청크가 하나도 나오지 않았다. 입력을 확인하라.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
