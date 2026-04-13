import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


class DataAccessAgent:
    """Handles corpus ingestion and normalized token generation."""

    SUPPORTED_EXTENSIONS = {".txt", ".csv", ".json", ".xml"}

    def load_text_file(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".txt":
            return self._load_txt(file_path)
        if suffix == ".csv":
            return self._load_csv(file_path)
        if suffix == ".json":
            return self._load_json(file_path)
        return self._load_xml(file_path)

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_text(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def get_clean_text_from_file(self, file_path: str) -> str:
        raw_text = self.load_text_file(file_path)
        return self.preprocess_text(raw_text)

    def get_tokens_from_file(self, file_path: str) -> list[str]:
        clean_text = self.get_clean_text_from_file(file_path)
        return self.tokenize_text(clean_text)

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _load_csv(self, file_path: str) -> str:
        lines: list[str] = []
        with open(file_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                lines.append(" ".join(row))
        return "\n".join(lines)

    def _load_json(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return json.dumps(payload, ensure_ascii=False)

    def _load_xml(self, file_path: str) -> str:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return " ".join(
            node.strip()
            for node in root.itertext()
            if isinstance(node, str) and node.strip()
        )
