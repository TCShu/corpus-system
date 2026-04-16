from __future__ import annotations

import json
import re

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
    def kwic_analysis(keyword: str, window_size: int = 5, max_results: int = 50) -> str:
        """Run Keyword-in-Context (KWIC) concordance analysis.

        Args:
            keyword: The word to search for in the corpus.
            window_size: Number of words to show on each side of the keyword.
            max_results: Maximum number of concordance lines to return.
        """
        keyword = keyword.lower().strip()
        matches = []
        for index, token in enumerate(tokens):
            if token != keyword:
                continue
            left = " ".join(tokens[max(0, index - window_size): index])
            right = " ".join(tokens[index + 1: index + 1 + window_size])
            matches.append({
                "position": index,
                "left_context": left,
                "keyword": keyword,
                "right_context": right,
            })
            if len(matches) >= max_results:
                break

        result = {
            "analysis_type": "kwic",
            "keyword": keyword,
            "window_size": window_size,
            "matches": matches,
        }
        return json.dumps(result)

    return [kwic_analysis]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the KWIC (Keyword-in-Context) Concordance Agent for ACAS.
Your job is to find all occurrences of a keyword in the corpus and show surrounding context.
When given a query, extract the keyword to search for and call the kwic_analysis tool.
Always return the raw JSON result from the tool without modification."""


class KWICAgent:
    def __init__(self):
        self.llm = ChatOllama(model=get_model())

    def analyze(self, tokens: list[str], keyword: str, window_size: int = 5, max_results: int = 50) -> dict:
        tools = _make_tools(tokens)
        agent = create_react_agent(self.llm, tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Run KWIC concordance for keyword '{keyword}' with window size {window_size}, max {max_results} results."),
        ]

        try:
            result = agent.invoke({"messages": messages})
            last_message = result["messages"][-1].content
        except Exception:
            return self._fallback(tokens, keyword, window_size, max_results)

        try:
            return json.loads(last_message)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._fallback(tokens, keyword, window_size, max_results)

    def _fallback(self, tokens: list[str], keyword: str, window_size: int, max_results: int) -> dict:
        keyword = keyword.lower().strip()
        matches = []
        for index, token in enumerate(tokens):
            if token != keyword:
                continue
            left = " ".join(tokens[max(0, index - window_size): index])
            right = " ".join(tokens[index + 1: index + 1 + window_size])
            matches.append({"position": index, "left_context": left, "keyword": keyword, "right_context": right})
            if len(matches) >= max_results:
                break
        return {"analysis_type": "kwic", "keyword": keyword, "window_size": window_size, "matches": matches}
