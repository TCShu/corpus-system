from __future__ import annotations

import json
import math
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agents._shared import get_model

DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "were", "with",
}

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _make_tools(tokens: list[str]):
    @tool
    def compute_frequency(top_k: int = 20, exclude_stopwords: bool = True) -> str:
        """Compute word frequency distribution from the corpus tokens.

        Args:
            top_k: How many top words to return.
            exclude_stopwords: Whether to filter common stopwords.
        """
        filtered = [t for t in tokens if t not in DEFAULT_STOPWORDS] if exclude_stopwords else tokens
        counts = Counter(filtered)
        total = max(sum(counts.values()), 1)
        rows = [
            {
                "rank": rank,
                "word": word,
                "frequency": count,
                "relative_frequency": round(count / total, 6),
            }
            for rank, (word, count) in enumerate(counts.most_common(top_k), start=1)
        ]
        result = {
            "analysis_type": "frequency",
            "total_tokens": len(tokens),
            "rows": rows,
        }
        return json.dumps(result)

    return [compute_frequency]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Frequency Analysis Agent for ACAS (Academic Corpus Analysis System).
Your job is to compute word frequency distributions from corpus tokens.
When asked to perform frequency analysis, call the compute_frequency tool with appropriate parameters.
Always return the raw JSON result from the tool without modification."""


class FrequencyAgent:
    def __init__(self):
        self.llm = ChatOllama(model=get_model())

    def analyze(self, tokens: list[str], top_k: int = 20, exclude_stopwords: bool = True) -> dict:
        tools = _make_tools(tokens)
        agent = create_react_agent(self.llm, tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Run frequency analysis. Return top {top_k} words. Exclude stopwords: {exclude_stopwords}."),
        ]

        try:
            result = agent.invoke({"messages": messages})
            last_message = result["messages"][-1].content
        except Exception:
            return self._fallback(tokens, top_k, exclude_stopwords)

        try:
            return json.loads(last_message)
        except (json.JSONDecodeError, TypeError):
            import re
            match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._fallback(tokens, top_k, exclude_stopwords)

    def _fallback(self, tokens: list[str], top_k: int, exclude_stopwords: bool) -> dict:
        filtered = [t for t in tokens if t not in DEFAULT_STOPWORDS] if exclude_stopwords else tokens
        counts = Counter(filtered)
        total = max(sum(counts.values()), 1)
        return {
            "analysis_type": "frequency",
            "total_tokens": len(tokens),
            "rows": [
                {"rank": r, "word": w, "frequency": c, "relative_frequency": round(c / total, 6)}
                for r, (w, c) in enumerate(counts.most_common(top_k), start=1)
            ],
        }
