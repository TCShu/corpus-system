from __future__ import annotations

import json
import math
import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agents._shared import get_model


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _make_tools(tokens: list[str]):
    @tool
    def ngram_collocation_analysis(n_size: int = 2, min_frequency: int = 2, top_k: int = 20) -> str:
        """Run n-gram collocation analysis with PMI scoring.

        Args:
            n_size: Size of n-grams (2 = bigrams, 3 = trigrams, etc.).
            min_frequency: Minimum frequency to include an n-gram.
            top_k: How many top n-grams to return.
        """
        if n_size < 2:
            n_size = 2

        ngrams = [
            tuple(tokens[i: i + n_size])
            for i in range(max(len(tokens) - n_size + 1, 0))
        ]
        counts = Counter(ngrams)
        unigram_counts = Counter(tokens)
        token_total = max(len(tokens), 1)

        rows = []
        for ngram, freq in counts.most_common():
            if freq < min_frequency:
                continue
            p_ngram = freq / token_total
            p_parts = 1.0
            for part in ngram:
                p_parts *= unigram_counts[part] / token_total
            pmi = math.log2(p_ngram / p_parts) if p_parts > 0 else 0.0
            rows.append({
                "ngram_text": " ".join(ngram),
                "ngram_size": n_size,
                "frequency": freq,
                "pmi_score": round(pmi, 4),
            })
            if len(rows) >= top_k:
                break

        result = {
            "analysis_type": "ngram_collocation",
            "ngram_size": n_size,
            "rows": rows,
        }
        return json.dumps(result)

    return [ngram_collocation_analysis]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the N-gram Collocation Agent for ACAS.
Your job is to identify frequent word combinations and collocations in the corpus using PMI scoring.
When asked, determine the appropriate n-gram size (bigram=2, trigram=3) and call the ngram_collocation_analysis tool.
Always return the raw JSON result from the tool without modification."""


class NgramAgent:
    def __init__(self):
        self.llm = ChatOllama(model=get_model())

    def analyze(self, tokens: list[str], n_size: int = 2, min_frequency: int = 2, top_k: int = 20) -> dict:
        tools = _make_tools(tokens)
        agent = create_react_agent(self.llm, tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Run {n_size}-gram collocation analysis. Minimum frequency: {min_frequency}. Return top {top_k} results."),
        ]

        try:
            result = agent.invoke({"messages": messages})
            last_message = result["messages"][-1].content
        except Exception:
            return self._fallback(tokens, n_size, min_frequency, top_k)

        try:
            return json.loads(last_message)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._fallback(tokens, n_size, min_frequency, top_k)

    def _fallback(self, tokens: list[str], n_size: int, min_frequency: int, top_k: int) -> dict:
        if n_size < 2:
            n_size = 2
        ngrams = [tuple(tokens[i: i + n_size]) for i in range(max(len(tokens) - n_size + 1, 0))]
        counts = Counter(ngrams)
        unigram_counts = Counter(tokens)
        token_total = max(len(tokens), 1)
        rows = []
        for ngram, freq in counts.most_common():
            if freq < min_frequency:
                continue
            p_ngram = freq / token_total
            p_parts = 1.0
            for part in ngram:
                p_parts *= unigram_counts[part] / token_total
            pmi = math.log2(p_ngram / p_parts) if p_parts > 0 else 0.0
            rows.append({"ngram_text": " ".join(ngram), "ngram_size": n_size, "frequency": freq, "pmi_score": round(pmi, 4)})
            if len(rows) >= top_k:
                break
        return {"analysis_type": "ngram_collocation", "ngram_size": n_size, "rows": rows}
