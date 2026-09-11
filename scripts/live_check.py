"""실제 Gemini API 를 호출하는 확인 스크립트.

**CI 에서 돌지 않는다.** 단위 테스트는 공급사를 더블로 바꿔 응답을 고정하고, 이
스크립트만 실제 호출을 한다. 매 커밋마다 과금되거나, 무료 티어의 503 때문에 CI 가
빨갛게 되는 일을 막기 위해서다.

무료 티어는 입출력이 모델 개선에 사용될 수 있다. **공개 문서만 넣는다.**

    uv run python scripts/live_check.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from d0c_sight.llm.config import LlmConfig
from d0c_sight.llm.diagnose import DiagnosisError, diagnose
from d0c_sight.llm.gemini import GeminiProvider
from d0c_sight.llm.protocol import ProviderError
from d0c_sight.pipeline.run import read_jsonl

CHUNKS = Path("data/processed/chunks.jsonl")
ERROR = "PostgreSQL: FATAL: sorry, too many clients already"
PRIMARY = "runtime-config-connection#RUNTIME-CONFIG-CONNECTION-SETTINGS:2"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not CHUNKS.is_file():
        print(f"먼저 `d0c-sight ingest` 로 {CHUNKS} 를 만들어라.", file=sys.stderr)
        return 1
    cfg = LlmConfig.from_env()
    if not os.environ.get(cfg.api_key_env):
        print(f"{cfg.api_key_env} 가 없다. .env 를 로드했는지 확인하라.", file=sys.stderr)
        return 1

    by_id = {c.chunk_id: c for c in read_jsonl(CHUNKS)}
    if PRIMARY not in by_id:
        print(f"청크를 찾을 수 없다: {PRIMARY}", file=sys.stderr)
        return 1

    cases = [
        ("정상 문맥", [by_id[PRIMARY]]),
        ("빈 문맥", []),
    ]
    failures = 0
    for label, context in cases:
        try:
            result = diagnose(ERROR, context, GeminiProvider(cfg.api_key_env), cfg, by_id.keys())
        except (ProviderError, DiagnosisError) as exc:
            print(f"[{label}] 실패: {exc}")
            failures += 1
            continue
        rec = result.record
        print(
            f"[{label}] {rec.provider}/{rec.model} {rec.latency_ms}ms "
            f"stop={rec.stop_reason} in={rec.input_tokens} out={rec.output_tokens} "
            f"think={rec.thinking_tokens} 폴백={rec.fallback_count} "
            f"원인={len(result.diagnosis.causes)} "
            f"근거탈락={len(result.dropped_evidence)} "
            f"insufficient={result.diagnosis.insufficient_evidence}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
