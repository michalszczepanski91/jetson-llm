"""Exact-length prompts, built once and reused byte-for-byte everywhere.

Two requirements meet in this file and they pull in opposite directions.

**The prompt-length sweep needs exact token counts.** "TTFT at 2048 tokens"
is a claim about 2048 tokens; a prompt that happens to tokenise to 1900 on
one backend and 2100 on another turns a sweep into noise. `benchmarks/
runner.py`'s `build_prompt` only ever *approximated* a length (0.75
words-per-token) and said so - fine when the achieved count is read back and
used for every derived figure, useless when the length itself is the x axis.

**The cross-framework comparison needs byte-identical prompts.** The task
brief asks for byte equality asserted across the three frameworks, and this
repo has already spent a campaign proving the framework gap is not a
chat-template artifact (`scripts/chat_template_crossfeed.py`). If each
backend tokenised its own way to hit its own exact 2048, the prompts would
differ in bytes and that whole control would be undone.

The resolution: exactness is established ONCE, against one reference
tokenizer, and the resulting bytes are frozen into a corpus file that every
framework then sends verbatim. Each framework reports whatever
`prompt_tokens` its own tokenizer and chat template produce, and that
achieved number - not the target - is what every derived figure uses. So the
target is exact in the corpus, the bytes are identical across backends, and
the per-backend difference (chat-template overhead, a few dozen tokens)
becomes a recorded quantity instead of a hidden one.

The corpus is content-addressed by (reference model, template version,
targets) and its sha256 goes into every run, so a figure can always be
traced to the exact bytes that produced it - the same discipline
`PROMPT_TEMPLATE_VERSION` already enforces one level up.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

#: Bumped whenever the corpus generator's text changes. A prompt change is an
#: experimental change (docs/note.md §21), so it may not happen silently.
CORPUS_VERSION = "c1"

#: Vocabulary the filler is drawn from. Deliberately in-domain for this lab -
#: a home-robot perception transcript, the thing the orchestrator LLM actually
#: reads - rather than lorem ipsum: prefill cost is roughly length-driven, but
#: an out-of-domain prompt would make the *quality* side of any run that
#: reuses this corpus meaningless, and one corpus serving both is worth more
#: than two that cannot be compared.
_VOCAB = (
    "the camera observes a person near the kitchen counter holding a blue mug "
    "while the robot waits by the doorway and the microwave door stays open "
    "someone left a laptop on the dining table next to a stack of plates and "
    "two chairs were moved since the last observation the lighting is dim in "
    "the hallway but bright in the living room where a cat sleeps on the sofa "
    "a bag of groceries sits unopened on the floor beside the refrigerator and "
    "the window blinds are half drawn against the afternoon sun outside"
).split()

_QUESTION = "\n\nGiven the observations above, what should the robot do next?"


def _shuffled_filler(n_words: int, seed: int) -> str:
    """Filler that is deterministic but not a repeated cycle.

    `runner.py`'s `_FILLER` repeats one two-sentence block, which made a
    shorter cell's prompt a literal PREFIX of a longer cell's - the bug that
    contaminated a published context sweep progressively with length (see
    that module's `build_prompt` docstring). Drawing from a shuffled
    vocabulary with a length-dependent seed means the 512-token prompt is not
    a prefix of the 2048-token one, so no prefix cache can bridge two cells
    of this sweep even if a backend's caching is left on by accident."""
    rng = random.Random(seed)
    words = []
    while len(words) < n_words:
        block = list(_VOCAB)
        rng.shuffle(block)
        words.extend(block)
    return " ".join(words[:n_words])


def _candidate(target_tokens: int, n_words: int) -> str:
    return f"[ctx{target_tokens}] " + _shuffled_filler(n_words, seed=target_tokens) + _QUESTION


class TokenCounter:
    """Counts tokens for a candidate string, by whatever means the reference
    server offers.

    Preference order matters. `/tokenize` is exact and costs no generation;
    the usage-block fallback is equally exact but costs a (max_tokens=1)
    round trip per probe, and - critically - it measures content + chat
    template, not content alone. Which one was used is recorded, because the
    two answers differ by the template overhead and a reader comparing
    corpora built different ways needs to see that."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.method: str | None = None

    def _try_tokenize_endpoint(self, text: str) -> int | None:
        # Both spellings go in the same body on purpose. vLLM's /tokenize
        # reads `prompt`; llama-server's reads `content` and IGNORES an
        # unknown `prompt`, which is the whole defect this once caused:
        # sending only `prompt` made llama-server tokenise the empty string
        # and answer {"tokens": []}, a well-formed 200 meaning zero. Each
        # server takes the key it knows and ignores the other.
        req = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"model": self.model, "prompt": text,
                             "content": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None
        n: int | None = None
        if isinstance(body, dict):
            if isinstance(body.get("count"), int):
                n = body["count"]
            elif isinstance(body.get("tokens"), list):
                n = len(body["tokens"])
        # Zero tokens for non-empty text is not a measurement, it is a
        # misunderstood request body. Returning None here demotes this
        # counter to the usage-block fallback, which costs a round trip and
        # cannot be silently wrong in this direction.
        if n is not None and n <= 0 and text.strip():
            return None
        return n

    def _via_usage(self, text: str) -> int:
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1, "temperature": 0.0,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return body["usage"]["prompt_tokens"]

    def count(self, text: str) -> int:
        if self.method is None:
            n = self._try_tokenize_endpoint(text)
            self.method = "POST /tokenize (content only)" if n is not None else \
                          "usage.prompt_tokens (content + chat template)"
            if n is not None:
                return n
        if self.method.startswith("POST /tokenize"):
            n = self._try_tokenize_endpoint(text)
            if n is not None:
                return n
            self.method = "usage.prompt_tokens (content + chat template)"
        return self._via_usage(text)


def build_corpus(
    targets: list[int],
    counter: TokenCounter,
    *,
    max_iters: int = 12,
    tolerance: int = 0,
) -> dict[str, Any]:
    """Converge each target to an exact token count by adjusting word count.

    Newton-ish on a monotone integer function: measure, scale the word count
    by target/achieved, re-measure. Converges in 3-5 probes per target
    because tokens-per-word is nearly constant within one vocabulary. If it
    cannot hit the target exactly within `max_iters` it records the closest
    it reached and by how much it missed - an off-by-three prompt honestly
    labelled beats a silent one."""
    entries = {}
    for target in sorted(targets):
        n_words = max(1, int(target * 0.72))
        best: tuple[int, str, int] | None = None   # (|error|, text, achieved)
        probes = []
        for _ in range(max_iters):
            text = _candidate(target, n_words)
            achieved = counter.count(text)
            probes.append({"n_words": n_words, "achieved_tokens": achieved})
            err = abs(achieved - target)
            if best is None or err < best[0]:
                best = (err, text, achieved)
            if err <= tolerance:
                break
            # Guard the degenerate step: near convergence the ratio rounds to
            # 1.0 and the search would stall on the same word count forever.
            step = int(round(n_words * target / achieved)) if achieved else n_words + 1
            n_words = step if step != n_words else n_words + (1 if achieved < target else -1)
            n_words = max(1, n_words)
        assert best is not None
        err, text, achieved = best
        # An off-by-three prompt honestly labelled is fine. A prompt that
        # missed its target by a quarter is not a near-miss, it is a broken
        # counter - and the run that follows would spend 40 minutes
        # measuring prompt lengths nobody asked for, against an x axis
        # shared with the other frameworks. Die here instead, while the
        # only thing lost is a corpus build.
        if achieved <= 0 or err > max(4, target // 4):
            raise RuntimeError(
                f"corpus target {target} converged to {achieved} tokens "
                f"(error {achieved - target}) after {len(probes)} probes via "
                f"{counter.method!r}. That is a counting failure, not a near "
                f"miss - check the reference server's /tokenize contract "
                f"before measuring anything against this corpus."
            )
        entries[str(target)] = {
            "target_tokens": target,
            "reference_tokens": achieved,
            "exact": err == 0,
            "token_error": achieved - target,
            "n_probes": len(probes),
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
        }
    payload = {
        "corpus_version": CORPUS_VERSION,
        "reference_model": counter.model,
        "reference_count_method": counter.method,
        "targets": sorted(targets),
        "entries": entries,
    }
    payload["corpus_sha256"] = hashlib.sha256(
        json.dumps({k: v["sha256"] for k, v in entries.items()}, sort_keys=True).encode()
    ).hexdigest()
    return payload


def corpus_path(root: Path, reference_model: str) -> Path:
    slug = reference_model.replace("/", "--")
    return root / f"{slug}.{CORPUS_VERSION}.json"


def load_or_build(
    root: Path,
    targets: list[int],
    counter_factory: Callable[[], TokenCounter],
    reference_model: str,
) -> dict[str, Any]:
    """Load the cached corpus if it already covers every requested target,
    otherwise build and cache it.

    Reuse is the point: the corpus is the shared object that makes the three
    frameworks' prompts byte-identical, so rebuilding it per framework would
    quietly destroy the property it exists to provide. A cached corpus that
    is missing a target is REBUILT WHOLE rather than extended, so one file's
    entries always come from one generator run."""
    path = corpus_path(root, reference_model)
    if path.exists():
        cached = json.loads(path.read_text())
        if set(map(str, targets)) <= set(cached["entries"]):
            cached["_loaded_from"] = str(path)
            return cached
    built = build_corpus(targets, counter_factory())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(built, indent=2))
    built["_loaded_from"] = f"{path} (built now)"
    return built
