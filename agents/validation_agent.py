from __future__ import annotations

from typing import Any


class ValidationAgent:
    """WatchDog-style validation before results are shown to users."""

    def validate_result(self, result: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []

        if not isinstance(result, dict):
            return {
                "safe": False,
                "issues": ["Result payload is not a dictionary."],
                "warnings": [],
            }

        analysis_type = result.get("analysis_type")
        if not analysis_type:
            issues.append("Missing analysis_type field.")

        if analysis_type == "frequency":
            self._validate_frequency(result, issues)
        elif analysis_type == "kwic":
            self._validate_kwic(result, issues)
        elif analysis_type == "ngram_collocation":
            self._validate_ngram(result, issues)
        elif analysis_type == "keyword_comparison":
            self._validate_keyword(result, issues)
        elif analysis_type == "dynamic_code":
            self._validate_dynamic(result, issues, warnings)
        else:
            warnings.append(f"Unknown analysis type '{analysis_type}'.")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def _validate_frequency(self, result: dict[str, Any], issues: list[str]) -> None:
        rows = result.get("rows")
        if not isinstance(rows, list):
            issues.append("Frequency result rows must be a list.")
            return
        for row in rows:
            if row.get("frequency", 0) < 0:
                issues.append("Frequency cannot be negative.")

    def _validate_kwic(self, result: dict[str, Any], issues: list[str]) -> None:
        matches = result.get("matches")
        if not isinstance(matches, list):
            issues.append("KWIC matches must be a list.")
            return
        for match in matches:
            if "keyword" not in match:
                issues.append("KWIC match missing keyword.")
                break

    def _validate_ngram(self, result: dict[str, Any], issues: list[str]) -> None:
        rows = result.get("rows")
        if not isinstance(rows, list):
            issues.append("N-gram rows must be a list.")
            return
        for row in rows:
            if row.get("ngram_size", 0) < 2:
                issues.append("N-gram size must be >= 2.")
                break

    def _validate_keyword(self, result: dict[str, Any], issues: list[str]) -> None:
        rows = result.get("rows")
        if not isinstance(rows, list):
            issues.append("Keyword rows must be a list.")
            return
        for row in rows:
            if row.get("keyness_ratio", 0) < 0:
                issues.append("Keyness ratio cannot be negative.")
                break

    def _validate_dynamic(
        self,
        result: dict[str, Any],
        issues: list[str],
        warnings: list[str],
    ) -> None:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            issues.append("Dynamic metadata must be a dictionary.")
            return
        if not metadata.get("executed_in_container", False):
            issues.append("Dynamic code was not executed in a container.")
        if metadata.get("validator_notes"):
            warnings.append(metadata["validator_notes"])
