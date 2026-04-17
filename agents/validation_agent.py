"""
ValidationAgent — deterministic watchdog for ACAS analysis results.

Responsibilities:
  - Verify structural correctness (required fields, correct types)
  - Check data integrity (non-negative values, minimum sizes)
  - Confirm the result is safe and sensible to display
  - Return {safe, issues, warnings} without generating new content

No LLM inference is used. Validation is fully deterministic so it
cannot hallucinate, stall, or accidentally produce a second answer.
"""
from __future__ import annotations

from typing import Any


class ValidationAgent:

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
            return {
                "safe": False,
                "issues": ["Missing analysis_type field."],
                "warnings": [],
            }

        if analysis_type == "frequency":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Frequency result rows must be a list.")
            else:
                if not rows:
                    warnings.append("Frequency result contains no rows.")
                for row in rows:
                    if row.get("frequency", 0) < 0:
                        issues.append("Frequency value cannot be negative.")
                        break

        elif analysis_type == "kwic":
            matches = result.get("matches")
            if not isinstance(matches, list):
                issues.append("KWIC matches must be a list.")
            else:
                if not matches:
                    warnings.append("No concordance lines found for this keyword.")
                for match in matches:
                    if "keyword" not in match:
                        issues.append("KWIC match is missing the keyword field.")
                        break

        elif analysis_type == "ngram_collocation":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("N-gram rows must be a list.")
            else:
                if not rows:
                    warnings.append("No n-grams met the minimum frequency threshold.")
                for row in rows:
                    if row.get("ngram_size", 2) < 2:
                        issues.append("N-gram size must be at least 2.")
                        break

        elif analysis_type == "keyword_comparison":
            rows = result.get("rows")
            if not isinstance(rows, list):
                issues.append("Keyword comparison rows must be a list.")
            else:
                if not rows:
                    warnings.append("No distinctive keywords found between the two corpora.")
                for row in rows:
                    if row.get("keyness_ratio", 0) < 0:
                        issues.append("Keyness ratio cannot be negative.")
                        break

        elif analysis_type == "dynamic_code":
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                issues.append("Dynamic code metadata must be a dictionary.")
            else:
                if not metadata.get("executed_in_container", False):
                    issues.append(
                        "Dynamic code was not executed in a sandboxed container."
                    )
                if metadata.get("validator_notes"):
                    warnings.append(metadata["validator_notes"])

        elif analysis_type == "conversational":
            reply = result.get("reply", "")
            if not reply or not reply.strip():
                issues.append("Conversational result has an empty reply.")

        else:
            warnings.append(
                f"Unrecognised analysis type '{analysis_type}' — result not fully validated."
            )

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
