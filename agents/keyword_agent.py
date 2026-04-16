from __future__ import annotations

import json
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

def _make_tools(target_tokens: list[str], reference_tokens: list[str]):
    @tool
    def keyword_comparison_analysis(top_k: int = 20, min_frequency: int = 2) -> str:
        """Compare keyword distribution between target corpus and reference corpus using keyness ratio.

        Args:
            top_k: Number of top keywords to return.
            min_frequency: Minimum frequency in target corpus to consider a word.
        """
        target_counts = Counter(target_tokens)
        ref_counts = Counter(reference_tokens)
        target_total = max(len(target_tokens), 1)
        ref_total = max(len(reference_tokens), 1)
        epsilon = 1e-9

        rows = []
        for word, target_freq in target_counts.items():
            if target_freq < min_frequency:
                continue
            target_rate = target_freq / target_total
            ref_rate = ref_counts.get(word, 0) / ref_total
            keyness = (target_rate + epsilon) / (ref_rate + epsilon)
            rows.append({
                "word": word,
                "target_frequency": target_freq,
                "reference_frequency": ref_counts.get(word, 0),
                "keyness_ratio": round(keyness, 6),
            })

        rows.sort(key=lambda item: item["keyness_ratio"], reverse=True)
        result = {
            "analysis_type": "keyword_comparison",
            "rows": rows[:top_k],
        }
        return json.dumps(result)

    return [keyword_comparison_analysis]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Keyword Comparison Agent for ACAS.
Your job is to identify statistically distinctive keywords by comparing a target corpus against a reference corpus using keyness ratio scoring.
When asked, call the keyword_comparison_analysis tool.
Always return the raw JSON result from the tool without modification."""


class KeywordAgent:
    def __init__(self):
        self.llm = ChatOllama(model=get_model())

    def analyze(
        self,
        target_tokens: list[str],
        reference_tokens: list[str],
        top_k: int = 20,
        min_frequency: int = 2,
    ) -> dict:
        tools = _make_tools(target_tokens, reference_tokens)
        agent = create_react_agent(self.llm, tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Run keyword comparison analysis. Return top {top_k} keywords with minimum frequency {min_frequency}."),
        ]

        try:
            result = agent.invoke({"messages": messages})
            last_message = result["messages"][-1].content
        except Exception:
            return self._fallback(target_tokens, reference_tokens, top_k, min_frequency)

        try:
            return json.loads(last_message)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._fallback(target_tokens, reference_tokens, top_k, min_frequency)

    def _fallback(self, target_tokens: list[str], reference_tokens: list[str], top_k: int, min_frequency: int) -> dict:
        target_counts = Counter(target_tokens)
        ref_counts = Counter(reference_tokens)
        target_total = max(len(target_tokens), 1)
        ref_total = max(len(reference_tokens), 1)
        epsilon = 1e-9
        rows = []
        for word, target_freq in target_counts.items():
            if target_freq < min_frequency:
                continue
            target_rate = target_freq / target_total
            ref_rate = ref_counts.get(word, 0) / ref_total
            keyness = (target_rate + epsilon) / (ref_rate + epsilon)
            rows.append({"word": word, "target_frequency": target_freq, "reference_frequency": ref_counts.get(word, 0), "keyness_ratio": round(keyness, 6)})
        rows.sort(key=lambda item: item["keyness_ratio"], reverse=True)
        return {"analysis_type": "keyword_comparison", "rows": rows[:top_k]}
