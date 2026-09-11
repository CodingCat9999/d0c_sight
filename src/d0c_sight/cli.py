"""명령줄 진입점.

Phase 1 에서는 수집과 청킹만 한다. 검색도 LLM 도 없다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from d0c_sight.pipeline.run import chunk_documents, read_jsonl, write_jsonl
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

    dia = sub.add_parser("diagnose", help="에러 메시지를 진단한다 (검색 없음)")
    dia.add_argument("error", help="진단할 에러 메시지")
    dia.add_argument(
        "--chunk-id",
        action="append",
        default=[],
        metavar="ID",
        help="컨텍스트로 넣을 청크. 검색이 없으므로 직접 지정한다 (반복 가능)",
    )
    dia.add_argument("--chunks", type=Path, default=DEFAULT_OUT, help="청크 JSONL 경로")
    dia.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")

    ev = sub.add_parser("eval", help="평가셋을 실행한다 (정답 청크 주입, 폴백 없음)")
    ev.add_argument("--evalset", type=Path, default=Path("evalset/postgres-18.json"))
    ev.add_argument("--chunks", type=Path, default=DEFAULT_OUT)
    ev.add_argument("--model", default=None, help="이 실행에 고정할 모델")
    ev.add_argument("--split", choices=["improve", "holdout", "all"], default="improve")
    ev.add_argument("--out", type=Path, default=Path("evalset/runs"))
    ev.add_argument("--cache", type=Path, default=Path("evalset/cache"))
    ev.add_argument("--no-cache", action="store_true")
    ev.add_argument("--label", default="", help="결과 파일 이름에 붙일 꼬리표")
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


def _diagnose(args: argparse.Namespace) -> int:
    from d0c_sight.llm.config import LlmConfig
    from d0c_sight.llm.diagnose import DiagnosisError, diagnose
    from d0c_sight.llm.gemini import GeminiProvider
    from d0c_sight.llm.protocol import ProviderError

    if not args.chunks.is_file():
        print(f"청크 파일이 없다: {args.chunks}", file=sys.stderr)
        print("먼저 `d0c-sight ingest` 를 실행하라.", file=sys.stderr)
        return 1

    by_id = {c.chunk_id: c for c in read_jsonl(args.chunks)}
    wanted = list(args.chunk_id)
    missing = [i for i in wanted if i not in by_id]
    if missing:
        print(f"존재하지 않는 청크 id: {missing}", file=sys.stderr)
        return 1
    context = [by_id[i] for i in wanted]

    try:
        provider = GeminiProvider(api_key_env=LlmConfig().api_key_env)
        result = diagnose(
            args.error, context, provider, LlmConfig.from_env(), known_ids=by_id.keys()
        )
    except (ProviderError, DiagnosisError) as exc:
        print(f"진단 실패: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    _print_result(result, len(context))
    return 0


def _print_result(result: object, context_size: int) -> None:
    from d0c_sight.domain.models import DiagnosisResult

    assert isinstance(result, DiagnosisResult)
    d, rec = result.diagnosis, result.record

    print(f"기준: {d.version_basis}   컨텍스트 청크 {context_size}개")
    print(f"모델: {rec.provider}/{rec.model}  ({rec.latency_ms}ms)")
    if rec.fallback_count:
        print(f"  폴백 {rec.fallback_count}회 — 시도: {' → '.join(rec.attempted_models)}")
    print(
        f"토큰: in={rec.input_tokens} out={rec.output_tokens} "
        f"thinking={rec.thinking_tokens}  stop={rec.stop_reason}"
    )
    print()

    if d.insufficient_evidence:
        print("근거 부족 — 이 컨텍스트로는 답할 수 없다.")
    for i, cause in enumerate(d.causes, start=1):
        print(f"원인 후보 {i} [{cause.confidence.value}]")
        print(f"  {cause.description}")
        for ev in cause.evidence:
            print(f"  근거: {ev.chunk_id}")
            print(f'        "{ev.quote[:100]}"')
    if d.checks:
        print("\n확인할 것")
        for chk in d.checks:
            print(f"  - {chk}")

    if result.dropped_evidence:
        print(f"\n[한계] 검증에 실패해 버린 근거 {len(result.dropped_evidence)}건")
        for drop in result.dropped_evidence:
            print(f"  - {drop.reason.value}: {drop.chunk_id or '(id 없음)'}")


def _eval(args: argparse.Namespace) -> int:
    import logging
    from dataclasses import replace

    from d0c_sight.evalset import integrity, store
    from d0c_sight.evalset.models import Split
    from d0c_sight.evalset.run import ResultCache, run_evalset
    from d0c_sight.llm.config import LlmConfig
    from d0c_sight.llm.gemini import GeminiProvider
    from d0c_sight.llm.protocol import ProviderError

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.evalset.is_file():
        print(f"평가셋이 없다: {args.evalset}", file=sys.stderr)
        return 1
    if not args.chunks.is_file():
        print(f"청크가 없다: {args.chunks}. 먼저 ingest 를 실행하라.", file=sys.stderr)
        return 1

    evalset = store.load(args.evalset)
    chunks = {c.chunk_id: c for c in read_jsonl(args.chunks)}

    issues = integrity.check(evalset, chunks)
    if issues:
        print(f"[무결성] 문제 {len(issues)}건 — 실행 전에 확인하라", file=sys.stderr)
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        return 1

    cases = None if args.split == "all" else evalset.by_split(Split(args.split))
    cfg = LlmConfig()
    if args.model:
        cfg = replace(cfg, model=args.model)

    try:
        provider = GeminiProvider(api_key_env=cfg.api_key_env)
    except ProviderError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    cache = None if args.no_cache else ResultCache(args.cache)
    n = len(cases) if cases is not None else len(evalset.cases)
    print(
        f"실행: {n}문항  모델 {cfg.model}  split={args.split}  캐시={'off' if args.no_cache else 'on'}"
    )
    run = run_evalset(evalset, provider, chunks, cfg, cases, cache)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = run.started_at.replace(":", "").replace("-", "")
    tag = f"-{args.label}" if args.label else ""
    path = args.out / f"{stamp}-{cfg.model}-{args.split}{tag}.json"
    path.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n결과 {path}")
    print(
        f"  API 호출 {run.api_calls}회  {run.duration_s}s  캐시적중 "
        f"{sum(1 for c in run.cases if c.from_cache)}/{len(run.cases)}"
    )
    print(
        f"  완전성 {run.complete}"
        + (
            f"  ← 실패 {len(run.failed_case_ids)}문항: {', '.join(run.failed_case_ids)}"
            if not run.complete
            else ""
        )
    )
    rate = run.pass_rate
    print(
        f"  전 항목 통과: {'—  (불완전한 실행이라 계산하지 않는다)' if rate is None else f'{rate:.1%}'}"
    )
    print("\n  항목별")
    for name, (ok, total) in sorted(run.check_totals().items()):
        print(f"    {name:<22} {ok:>3}/{total:<3} {ok / total:.0%}")
    causes = [c.insufficient_cause for c in run.cases if c.insufficient_cause]
    if causes:
        from collections import Counter

        print(f"\n  근거 부족 사유: {dict(Counter(causes))}")
    return 0 if run.complete else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command == "eval":
        return _eval(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
