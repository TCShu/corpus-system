from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agents._shared import get_model


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _make_tools(result: dict[str, Any]):
    @tool
    def validate_analysis_result() -> str:
        """Validate the structure and integrity of an analysis result produced by a specialist agent.
        Checks for required fields, negative values, and structural correctness.
        """
        issues: list[str] = []
        warnings: list[str] = []

        if not isinstance(result, dict):
            return json.dumps({"safe": False, "issues": ["Result payload is not a dictionary."], "warnings": []})

        analysis_type = result.get("analysis_type")
        if not analysis_type:
            issues.append("Missing analysis_type field.")

        if analysis_type == "frequency":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Frequency result rows must be a list.")
            else:
                for row in rows:
                    if row.get("frequency", 0) < 0:
                        issues.append("Frequency cannot be negative.")

        elif analysis_type == "kwic":
            matches = result.get("matches")
            if not isinstance(matches, list):
                issues.append("KWIC matches must be a list.")
            else:
                for match in matches:
                    if "keyword" not in match:
                        issues.append("KWIC match missing keyword.")
                        break

        elif analysis_type == "ngram_collocation":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("N-gram rows must be a list.")
            else:
                for row in rows:
                    if row.get("ngram_size", 0) < 2:
                        issues.append("N-gram size must be >= 2.")
                        break

        elif analysis_type == "keyword_comparison":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Keyword rows must be a list.")
            else:
                for row in rows:
                    if row.get("keyness_ratio", 0) < 0:
                        issues.append("Keyness ratio cannot be negative.")
                        break

        elif analysis_type == "dynamic_code":
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                issues.append("Dynamic metadata must be a dictionary.")
            else:
                if not metadata.get("executed_in_container", False):
                    issues.append("Dynamic code was not executed in a container.")
                if metadata.get("validator_notes"):
                    warnings.append(metadata["validator_notes"])

        else:
            warnings.append(f"Unknown analysis type '{analysis_type}'.")

        return json.dumps({"safe": len(issues) == 0, "issues": issues, "warnings": warnings})

    return [validate_analysis_result]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Validation Agent for ACAS (WatchDog).
Your job is to validate analysis results produced by other agents before they are shown to users.
You check for structural correctness, missing fields, and data integrity violations.
Always call the validate_analysis_result tool and return its JSON output exactly."""


class ValidationAgent:
    def __init__(self):
        self.llm = ChatOllama(model=get_model())

    def validate_result(self, result: dict[str, Any]) -> dict[str, Any]:
        tools = _make_tools(result)
        agent = create_react_agent(self.llm, tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Validate this analysis result and return the validation report: {json.dumps(result)[:500]}"),
        ]

        try:
            agent_result = agent.invoke({"messages": messages})
            last_message = agent_result["messages"][-1].content
        except Exception:
            return self._fallback_validate(result)

        try:
            return json.loads(last_message)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._fallback_validate(result)

    def _fallback_validate(self, result: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []

        if not isinstance(result, dict):
            return {"safe": False, "issues": ["Result payload is not a dictionary."], "warnings": []}

        analysis_type = result.get("analysis_type")
        if not analysis_type:
            issues.append("Missing analysis_type field.")

        if analysis_type == "frequency":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Frequency result rows must be a list.")
            else:
                for row in rows:
                    if row.get("frequency", 0) < 0:
                        issues.append("Frequency cannot be negative.")

        elif analysis_type == "kwic":
            matches = result.get("matches")
            if not isinstance(matches, list):
                issues.append("KWIC matches must be a list.")
            else:
                for match in matches:
                    if "keyword" not in match:
                        issues.append("KWIC match missing keyword.")
                        break

        elif analysis_type == "ngram_collocation":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("N-gram rows must be a list.")
            else:
                for row in rows:
                    if row.get("ngram_size", 0) < 2:
                        issues.append("N-gram size must be >= 2.")
                        break

        elif analysis_type == "keyword_comparison":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Keyword rows must be a list.")
            else:
                for row in rows:
                    if row.get("keyness_ratio", 0) < 0:
                        issues.append("Keyness ratio cannot be negative.")
                        break

        elif analysis_type == "dynamic_code":
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                issues.append("Dynamic metadata must be a dictionary.")
            else:
                if not metadata.get("executed_in_container", False):
                    issues.append("Dynamic code was not executed in a container.")
                if metadata.get("validator_notes"):
                    warnings.append(metadata["validator_notes"])

        else:
            warnings.append(f"Unknown analysis type '{analysis_type}'.")

        return {"safe": len(issues) == 0, "issues": issues, "warnings": warnings}
