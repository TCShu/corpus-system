"""
ACAS Agent Performance Benchmark
==================================
Measures end-to-end response time for each agent route through the full
CoordinationAgent graph (router → specialist → validator).

Also benchmarks each specialist agent directly so LLM routing overhead
can be separated from the analysis computation itself.

Usage:
    python benchmarks/benchmark_agents.py [--corpus path/to/corpus.txt] [--runs 3]

Corpus defaults to a 10,000-word synthetic corpus when no file is provided.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.coordination_agent import CoordinationAgent
from agents.frequency_agent import FrequencyAgent
from agents.keyword_agent import KeywordAgent
from agents.kwic_agent import KWICAgent
from agents.ngram_agent import NgramAgent


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _load_corpus(path: str | None) -> list[str]:
    if path:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        tokens = text.lower().split()
        print(f"Loaded corpus from {path!r}: {len(tokens):,} tokens")
        return tokens
    return _synthetic_corpus(10_000)


def _synthetic_corpus(n: int) -> list[str]:
    """Build a deterministic synthetic corpus of *n* tokens."""
    import random
    rng = random.Random(42)
    # Word pool drawn from typical academic English
    pool = [
        "the", "of", "and", "in", "to", "a", "is", "that", "for", "it",
        "with", "as", "was", "on", "are", "be", "by", "this", "an", "from",
        "analysis", "corpus", "language", "data", "text", "word", "frequency",
        "research", "study", "method", "result", "approach", "model", "term",
        "context", "discourse", "semantic", "lexical", "syntax", "grammar",
        "academic", "linguistic", "pattern", "structure", "distribution",
        "hypothesis", "evidence", "significant", "finding", "sample",
        "university", "paper", "journal", "author", "citation", "reference",
    ]
    tokens = [rng.choice(pool) for _ in range(n)]
    print(f"Generated synthetic corpus: {len(tokens):,} tokens")
    return tokens


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs) -> tuple[float, object]:
    """Run *fn* and return (elapsed_seconds, return_value)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def _run_n(label: str, fn, n: int, *args, **kwargs) -> list[float]:
    """Run *fn* n times, print each result, and return the list of durations."""
    times: list[float] = []
    for i in range(1, n + 1):
        elapsed, _ = _timed(fn, *args, **kwargs)
        times.append(elapsed)
        print(f"  [{label}] run {i}/{n}: {elapsed:.2f}s")
    return times


def _report(label: str, times: list[float]) -> None:
    avg = statistics.mean(times)
    low = min(times)
    high = max(times)
    med = statistics.median(times)
    print(f"\n  {label}")
    print(f"    avg={avg:.2f}s  median={med:.2f}s  min={low:.2f}s  max={high:.2f}s  (n={len(times)})")


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def bench_coordinator(tokens: list[str], reference_tokens: list[str], runs: int) -> None:
    print("\n" + "=" * 60)
    print("SUITE 1 — Full pipeline via CoordinationAgent")
    print("=" * 60)
    print("Initialising CoordinationAgent (first call may be slower)…")
    coordinator = CoordinationAgent()
    n = len(tokens)

    queries = [
        ("frequency", "Show me the top 20 most frequent words"),
        ("kwic",      "Show KWIC concordance lines for 'language'"),
        ("ngram",     "What are the most common bigrams?"),
        ("keyword",   "Compare keywords against the reference corpus"),
    ]

    results: dict[str, list[float]] = {}
    for route, query in queries:
        print(f"\n--- {route.upper()} ({n:,}-token corpus) ---")
        ref = reference_tokens if route == "keyword" else None
        times = _run_n(
            route, coordinator.execute,
            runs,
            query=query,
            tokens=tokens,
            reference_tokens=ref,
        )
        results[route] = times

    print("\n" + "-" * 60)
    print(f"COORDINATOR SUMMARY  ({n:,}-word corpus, {runs} run(s) each)")
    print("-" * 60)
    for route, times in results.items():
        _report(route, times)


def bench_specialists(tokens: list[str], reference_tokens: list[str], runs: int) -> None:
    print("\n" + "=" * 60)
    print("SUITE 2 — Specialist agents called directly (no routing LLM)")
    print("=" * 60)
    n = len(tokens)

    freq_agent  = FrequencyAgent()
    kwic_agent  = KWICAgent()
    ngram_agent = NgramAgent()
    kw_agent    = KeywordAgent()

    tasks = [
        (
            "FrequencyAgent",
            freq_agent.analyze,
            dict(tokens=tokens, top_k=20, exclude_stopwords=True),
        ),
        (
            "KWICAgent",
            kwic_agent.analyze,
            dict(tokens=tokens, keyword="language", window_size=5, max_results=50),
        ),
        (
            "NgramAgent",
            ngram_agent.analyze,
            dict(tokens=tokens, n_size=2, min_frequency=2, top_k=20),
        ),
        (
            "KeywordAgent",
            kw_agent.analyze,
            dict(target_tokens=tokens, reference_tokens=reference_tokens, top_k=20, min_frequency=2),
        ),
    ]

    results: dict[str, list[float]] = {}
    for label, fn, kwargs in tasks:
        print(f"\n--- {label} ({n:,}-token corpus) ---")
        times = _run_n(label, fn, runs, **kwargs)
        results[label] = times

    print("\n" + "-" * 60)
    print(f"SPECIALIST SUMMARY  ({n:,}-word corpus, {runs} run(s) each)")
    print("-" * 60)
    for label, times in results.items():
        _report(label, times)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ACAS agent performance benchmark")
    parser.add_argument("--corpus",    default=None, help="Path to a plain-text corpus file")
    parser.add_argument("--reference", default=None, help="Path to a reference corpus file (for keyword agent)")
    parser.add_argument("--runs",      type=int, default=3, help="Number of timed runs per agent (default: 3)")
    parser.add_argument("--suite",     choices=["all", "coordinator", "specialists"], default="all",
                        help="Which benchmark suite to run (default: all)")
    args = parser.parse_args()

    tokens           = _load_corpus(args.corpus)
    reference_tokens = _load_corpus(args.reference) if args.reference else _synthetic_corpus(5_000)
    if not args.reference:
        print("No reference corpus provided — using a 5,000-token synthetic corpus for keyword comparisons.")

    if args.suite in ("all", "coordinator"):
        bench_coordinator(tokens, reference_tokens, args.runs)

    if args.suite in ("all", "specialists"):
        bench_specialists(tokens, reference_tokens, args.runs)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
