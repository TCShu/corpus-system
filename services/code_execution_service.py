from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class SafeCodeExecutionService:
    """Executes generated Python only inside a locked-down Docker container."""

    BANNED_PATTERNS = [
        r"\bimport\s+os\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+socket\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\b__import__\s*\(",
    ]

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        issues: list[str] = []
        for pattern in self.BANNED_PATTERNS:
            if re.search(pattern, code):
                issues.append(f"Blocked pattern detected: {pattern}")
        if "result =" not in code:
            issues.append("Dynamic code must assign final output to `result`.")
        return len(issues) == 0, issues

    def execute(self, code: str, timeout_seconds: int = 20) -> dict[str, Any]:
        is_valid, issues = self.validate_code(code)
        if not is_valid:
            return {
                "analysis_type": "dynamic_code",
                "result": None,
                "metadata": {
                    "executed_in_container": False,
                    "errors": issues,
                    "validator_notes": "Code failed static safety checks.",
                },
            }

        wrapper = (
            "import json\n"
            "result = None\n"
            f"{code}\n"
            "print(json.dumps({'result': result}, ensure_ascii=False))\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "task.py"
            script_path.write_text(wrapper, encoding="utf-8")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                "0.5",
                "--memory",
                "256m",
                "-v",
                f"{script_path}:/app/task.py:ro",
                "--read-only",
                "python:3.11-slim",
                "python",
                "/app/task.py",
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except FileNotFoundError:
                return {
                    "analysis_type": "dynamic_code",
                    "result": None,
                    "metadata": {
                        "executed_in_container": False,
                        "errors": ["Docker is not installed or not available in PATH."],
                        "validator_notes": "Container execution was required but unavailable.",
                    },
                }
            except subprocess.TimeoutExpired:
                return {
                    "analysis_type": "dynamic_code",
                    "result": None,
                    "metadata": {
                        "executed_in_container": True,
                        "errors": ["Dynamic execution timed out."],
                        "validator_notes": "Execution exceeded timeout.",
                    },
                }

        if completed.returncode != 0:
            return {
                "analysis_type": "dynamic_code",
                "result": None,
                "metadata": {
                    "executed_in_container": True,
                    "errors": [completed.stderr.strip() or "Unknown execution error."],
                    "validator_notes": "Container execution failed.",
                },
            }

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            payload = {"result": completed.stdout.strip()}

        return {
            "analysis_type": "dynamic_code",
            "result": payload.get("result"),
            "metadata": {
                "executed_in_container": True,
                "errors": [],
                "validator_notes": "Executed in restricted container successfully.",
            },
        }
