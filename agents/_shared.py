"""Shared utilities for all ACAS AI agents."""
import os

AVAILABLE_MODELS = ["llama3", "gemma3:4b", "gemma3:12b", "gemma3:27b"]
_current_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")


def get_model() -> str:
    return _current_model


def set_model(model: str) -> None:
    global _current_model
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {AVAILABLE_MODELS}")
    _current_model = model
