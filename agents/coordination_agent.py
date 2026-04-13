from __future__ import annotations

import re
from typing import Any

from agents.frequency_agent import FrequencyAgent
from agents.keyword_agent import KeywordAgent
from agents.kwic_agent import KWICAgent
from agents.ngram_agent import NgramAgent
from agents.validation_agent import ValidationAgent
from services.code_execution_service import SafeCodeExecutionService


class CoordinationAgent:
    """Interprets natural language and dispatches the right analysis agent."""

    def __init__(self) -> None:
        self.frequency_agent = FrequencyAgent()
        self.kwic_agent = KWICAgent()
        self.ngram_agent = NgramAgent()
        self.keyword_agent = KeywordAgent()
        self.validation_agent = ValidationAgent()
        self.code_runner = SafeCodeExecutionService()

    def execute(
        self,
        query: str,
        tokens: list[str],
        reference_tokens: list[str] | None = None,
        corpus_text: str = "",
    ) -> dict[str, Any]:
        normalized = query.lower().strip()

        if self._is_frequency_request(normalized):
            top_k = self._extract_int(normalized, default=20)
            result = self.frequency_agent.analyze(tokens=tokens, top_k=top_k)
        elif self._is_kwic_request(normalized):
            keyword = self._extract_keyword(normalized)
            window = self._extract_window_size(normalized)
            result = self.kwic_agent.analyze(tokens=tokens, keyword=keyword, window_size=window)
        elif self._is_ngram_request(normalized):
            n_value = self._extract_ngram_size(normalized)
            result = self.ngram_agent.analyze(tokens=tokens, n_size=n_value)
        elif self._is_keyword_comparison_request(normalized):
            if not reference_tokens:
                return {
                    "safe": False,
                    "validation": {
                        "safe": False,
                        "issues": ["Keyword analysis requires a reference corpus."],
                        "warnings": [],
                    },
                    "result": None,
                }
            result = self.keyword_agent.analyze(
                target_tokens=tokens,
                reference_tokens=reference_tokens,
            )
        elif self._looks_linguistic(normalized):
            dynamic_code = self._build_dynamic_linguistic_snippet(normalized, corpus_text)
            result = self.code_runner.execute(dynamic_code)
        else:
            return {
                "safe": False,
                "validation": {
                    "safe": False,
                    "issues": [
                        "Query is out of ACAS scope. Please ask for corpus linguistic analysis."
                    ],
                    "warnings": [],
                },
                "result": None,
            }

        validation = self.validation_agent.validate_result(result)
        return {
            "safe": validation["safe"],
            "validation": validation,
            "result": result,
        }

    def _is_frequency_request(self, query: str) -> bool:
        return "frequency" in query or "frequent words" in query

    def _is_kwic_request(self, query: str) -> bool:
        return "kwic" in query or "concordance" in query or "context" in query

    def _is_ngram_request(self, query: str) -> bool:
        return "ngram" in query or "n-gram" in query or "collocation" in query or "bigram" in query or "trigram" in query

    def _is_keyword_comparison_request(self, query: str) -> bool:
        return "keyword" in query and "reference" in query or "compare corpus" in query

    def _looks_linguistic(self, query: str) -> bool:
        lexical_triggers = [
            "token",
            "lemma",
            "pos",
            "part of speech",
            "syntax",
            "semantic",
            "phrase",
            "linguistic",
            "corpus",
        ]
        return any(trigger in query for trigger in lexical_triggers)

    def _extract_int(self, query: str, default: int) -> int:
        match = re.search(r"\b(\d+)\b", query)
        return int(match.group(1)) if match else default

    def _extract_window_size(self, query: str) -> int:
        match = re.search(r"(\d+)\s*[- ]?word", query)
        return int(match.group(1)) if match else 5

    def _extract_ngram_size(self, query: str) -> int:
        if "trigram" in query:
            return 3
        if "bigram" in query:
            return 2
        match = re.search(r"(\d+)\s*[- ]?gram", query)
        if match:
            return max(int(match.group(1)), 2)
        return 2

    def _extract_keyword(self, query: str) -> str:
        quoted = re.search(r"['\"]([^'\"]+)['\"]", query)
        if quoted:
            return quoted.group(1).strip().lower()

        tokenized = re.findall(r"[a-zA-Z']+", query)
        fallback = tokenized[-1] if tokenized else ""
        return fallback.lower()

    def _build_dynamic_linguistic_snippet(self, query: str, corpus_text: str) -> str:
        # Kept intentionally constrained: only pure text computations.
        safe_text = corpus_text.replace("\\", "\\\\").replace("'", "\\'")
        safe_query = query.replace("\\", "\\\\").replace("'", "\\'")
        return (
            f"text = '{safe_text[:50000]}'\n"
            f"query = '{safe_query}'\n"
            "tokens = [t for t in text.lower().split() if t]\n"
            "unique_tokens = len(set(tokens))\n"
            "result = {\n"
            "  'query_interpreted_as': query,\n"
            "  'token_count': len(tokens),\n"
            "  'unique_token_count': unique_tokens,\n"
            "  'type_token_ratio': (unique_tokens / len(tokens)) if tokens else 0.0\n"
            "}\n"
        )
