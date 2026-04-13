from collections import Counter


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


class FrequencyAgent:
    def analyze(
        self,
        tokens: list[str],
        top_k: int = 20,
        exclude_stopwords: bool = True,
    ) -> dict:
        filtered = tokens
        if exclude_stopwords:
            filtered = [token for token in tokens if token not in DEFAULT_STOPWORDS]

        counts = Counter(filtered)
        total = max(sum(counts.values()), 1)
        rows = []
        for rank, (word, count) in enumerate(counts.most_common(top_k), start=1):
            rows.append(
                {
                    "rank": rank,
                    "word": word,
                    "frequency": count,
                    "relative_frequency": round(count / total, 6),
                }
            )

        return {
            "analysis_type": "frequency",
            "total_tokens": len(tokens),
            "rows": rows,
        }
