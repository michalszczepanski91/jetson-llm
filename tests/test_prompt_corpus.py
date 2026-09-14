"""The corpus is what makes three backends comparable, so its counter has
to fail loudly rather than quietly.

Written 2026-09-14 after a real run: `TokenCounter` posted only `prompt` to
/tokenize, llama-server reads `content`, and the well-formed 200 that came
back meant zero tokens. Every entry converged to `reference_tokens: 0`, the
max-context probe then asked for a 764,533-token prompt, and a 40-minute
llama.cpp run measured prompt lengths of 158/437/1555/6026 against the
vLLM leg's 173/555/2093/8239 - an x axis the two legs did not share, in a
comparison whose entire premise is that they do.

These tests run against a real local HTTP server rather than a patched
urlopen, because the defect lived in the request body: a mock that accepts
whatever the code sends could not have caught it.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

import prompts as pr     # noqa: E402


class _Server:
    """A tokenizer server with one knob: which request key it reads."""

    def __init__(self, reads: str, *, tokens_per_word: float = 1.0):
        self.reads = reads
        self.tokens_per_word = tokens_per_word
        self.bodies: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                if self.path.endswith("/tokenize"):
                    text = body.get(outer.reads) or ""
                    n = int(len(text.split()) * outer.tokens_per_word)
                    payload = {"tokens": list(range(n))}
                else:  # /v1/chat/completions - the usage fallback
                    text = body["messages"][0]["content"]
                    n = int(len(text.split()) * outer.tokens_per_word)
                    payload = {"choices": [{"message": {"content": "x"}}],
                               "usage": {"prompt_tokens": n}}
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def test_tokenize_body_carries_both_spellings():
    """vLLM reads `prompt`, llama-server reads `content`. One body, both
    keys - each server takes the one it knows and ignores the other."""
    with _Server(reads="prompt") as srv:
        pr.TokenCounter(srv.url, "m").count("one two three")
    body = srv.bodies[0]
    assert body["prompt"] == "one two three"
    assert body["content"] == "one two three"


def test_a_server_that_reads_only_content_is_counted_correctly():
    """The actual regression: llama-server's contract."""
    with _Server(reads="content") as srv:
        assert pr.TokenCounter(srv.url, "m").count("one two three four") == 4


def test_zero_tokens_for_nonempty_text_falls_back_instead_of_being_believed():
    """A 200 carrying {"tokens": []} is a misunderstood request body, not a
    measurement of zero - so the counter must demote itself to the usage
    block rather than return 0."""
    with _Server(reads="__nothing_reads_this__") as srv:
        counter = pr.TokenCounter(srv.url, "m")
        assert counter.count("one two three") == 3          # via the fallback
        assert counter.method.startswith("usage.prompt_tokens")


def test_empty_text_may_legitimately_count_zero():
    """The guard above must not make a genuinely empty prompt unmeasurable."""
    with _Server(reads="content") as srv:
        assert pr.TokenCounter(srv.url, "m").count("") == 0


def test_build_corpus_refuses_a_counter_that_cannot_hit_its_targets():
    class Dead:
        model, method = "m", "stub"

        def count(self, text):
            return 0

    with pytest.raises(RuntimeError, match="counting failure"):
        pr.build_corpus([128], Dead())


def test_build_corpus_still_accepts_an_honest_near_miss():
    """An off-by-a-few prompt honestly labelled is the documented behaviour
    and must survive the gate that rejects a broken counter."""
    class Coarse:
        model, method = "m", "stub"

        def count(self, text):           # quantised to 10s: cannot hit 128
            return round(len(text.split()) / 10) * 10

    corpus = pr.build_corpus([128], Coarse())
    entry = corpus["entries"]["128"]
    assert entry["exact"] is False
    assert abs(entry["token_error"]) <= max(4, 128 // 4)


def test_one_corpus_serves_every_framework_that_names_it(tmp_path):
    """load_or_build is keyed by reference model, and that is what keeps the
    three backends' prompts byte-identical. A row whose own `model` differs
    (a -GGUF repo) shares the corpus by naming the reference id."""
    with _Server(reads="content") as srv:
        first = pr.load_or_build(tmp_path, [128],
                                 lambda: pr.TokenCounter(srv.url, "Qwen/X"),
                                 reference_model="Qwen/X")
        calls_after_build = len(srv.bodies)
        second = pr.load_or_build(tmp_path, [128],
                                  lambda: pr.TokenCounter(srv.url, "Qwen/X-GGUF"),
                                  reference_model="Qwen/X")

    assert second["corpus_sha256"] == first["corpus_sha256"]
    assert second["entries"]["128"]["sha256"] == first["entries"]["128"]["sha256"]
    # The second caller read the file; it did not tokenise anything of its
    # own. If it had, the bytes could differ and the comparison would be
    # confounded by the thing the corpus exists to hold fixed.
    assert len(srv.bodies) == calls_after_build
