"""라벨링 도구 서버 — 표준 라이브러리만 쓴다.

개발자용 내부 도구다. 인증도 배포도 없고 로컬에서만 돈다. Phase 10 의 사용자
인터페이스와 코드를 공유하지 않는다 — 용도가 다르고, 여기 시간을 쓸 이유가 없다.

    uv run python tools/labeler/server.py

의존성 0 인 이유는 이 도구가 버려질 것이기 때문이다. 디렉터리 하나를 지우면 끝나야
하고, 본체의 `dependencies` 에 흔적이 남으면 안 된다.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from d0c_sight.evalset import integrity, store
from d0c_sight.evalset.models import EvalSet, content_hash
from d0c_sight.llm.verify import MIN_QUOTE_CHARS, can_be_quoted
from d0c_sight.pipeline.run import read_jsonl

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EVALSET_PATH = Path("evalset/postgres-18.json")
HUMAN_SCORES_PATH = Path("evalset/human_scores.json")
RUNS_DIR = Path("evalset/runs")
PORT = 8765

#: 검색 결과를 이만큼만 돌려준다. 16,442개를 브라우저에 밀어넣지 않는다.
SEARCH_LIMIT = 60


def _load_chunks() -> dict[str, Any]:
    if not CHUNKS_PATH.is_file():
        return {}
    return {c.chunk_id: c for c in read_jsonl(CHUNKS_PATH)}


class Store:
    """청크와 평가셋을 메모리에 들고 있는다. 단일 사용자용이므로 락이 필요 없다."""

    def __init__(self) -> None:
        self.chunks = _load_chunks()
        self.evalset = (
            store.load(EVALSET_PATH)
            if EVALSET_PATH.is_file()
            else EvalSet(version="v1", created="", corpus_version="18.6")
        )

    def chunk_view(self, chunk_id: str) -> dict[str, Any] | None:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None
        body = chunk.quotable
        return {
            "chunk_id": chunk.chunk_id,
            "kind": chunk.kind.value,
            "subtype": chunk.subtype,
            "heading": chunk.heading,
            "text": chunk.text,
            "body": body,
            "length": len(body),
            "url": chunk.ref.url,
            "doc_id": chunk.ref.doc_id,
            "section_id": chunk.ref.section_id,
            "version": chunk.ref.version.full,
            "part": chunk.part,
            "of": chunk.of,
            "content_sha256": content_hash(body),
            # 경고는 하되 막지 않는다. 회피하게 만들면 임계값이 부적절해도
            # 영원히 드러나지 않는다 — 이 값 자체가 Phase 3 의 측정 대상이다.
            "too_short_to_quote": not can_be_quoted(body),
            "min_quote_chars": MIN_QUOTE_CHARS,
        }

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[dict[str, Any]]:
        """정확 매칭 검색. 의미 검색은 Phase 4 이후다."""
        needle = query.strip().lower()
        if not needle:
            return []
        out: list[dict[str, Any]] = []
        for chunk in self.chunks.values():
            if needle in chunk.quotable.lower() or needle in chunk.chunk_id.lower():
                view = self.chunk_view(chunk.chunk_id)
                if view is not None:
                    out.append(view)
            if len(out) >= limit:
                break
        return out


STATE = Store()


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if not length:
        return {}
    parsed: dict[str, Any] = json.loads(handler.rfile.read(length))
    return parsed


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # 요청마다 찍히는 기본 로그가 터미널을 덮는다

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = (ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = parse_qs(url.query)
        if url.path in {"/", "/index.html"}:
            self._send_html()
        elif url.path == "/api/stats":
            self._send(
                {
                    "chunks": len(STATE.chunks),
                    "cases": len(STATE.evalset.cases),
                    "evalset_version": STATE.evalset.version,
                    "corpus_version": STATE.evalset.corpus_version,
                    "min_quote_chars": MIN_QUOTE_CHARS,
                }
            )
        elif url.path == "/api/search":
            self._send(STATE.search(params.get("q", [""])[0]))
        elif url.path == "/api/chunk":
            view = STATE.chunk_view(params.get("id", [""])[0])
            self._send(view if view else {"error": "not found"}, 200 if view else 404)
        elif url.path == "/api/evalset":
            self._send(asdict(STATE.evalset))
        elif url.path == "/api/integrity":
            issues = integrity.check(STATE.evalset, STATE.chunks)
            self._send([asdict(i) for i in issues])
        elif url.path == "/api/human-scores":
            self._send(
                json.loads(HUMAN_SCORES_PATH.read_text(encoding="utf-8"))
                if HUMAN_SCORES_PATH.is_file()
                else {}
            )
        elif url.path == "/api/runs":
            self._send(sorted(p.stem for p in RUNS_DIR.glob("*.json")))
        elif url.path == "/api/run":
            name = params.get("id", [""])[0]
            path = RUNS_DIR / f"{name}.json"
            if not path.is_file() or path.parent != RUNS_DIR:
                self._send({"error": "not found"}, 404)
            else:
                self._send(json.loads(path.read_text(encoding="utf-8")))
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        payload = _json_body(self)
        if url.path == "/api/evalset":
            try:
                STATE.evalset = store.load_dict(payload)
            except (ValueError, KeyError) as exc:
                self._send({"error": str(exc)}, 400)
                return
            store.save(STATE.evalset, EVALSET_PATH)
            self._send({"saved": len(STATE.evalset.cases), "path": str(EVALSET_PATH)})
        elif url.path == "/api/human-scores":
            HUMAN_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
            HUMAN_SCORES_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self._send({"saved": len(payload)})
        else:
            self._send({"error": "not found"}, 404)


def main() -> int:
    if not STATE.chunks:
        print(f"청크가 없다: {CHUNKS_PATH}. 먼저 `d0c-sight ingest` 를 실행하라.")
        return 1
    print(f"청크 {len(STATE.chunks):,}개  문항 {len(STATE.evalset.cases)}개")
    print(f"http://127.0.0.1:{PORT}  (Ctrl+C 로 종료)")
    import os

    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        if not os.environ.get("D0C_NO_BROWSER"):
            webbrowser.open(f"http://127.0.0.1:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
