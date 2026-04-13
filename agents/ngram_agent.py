import math
from collections import Counter


class NgramAgent:
    def analyze(
        self,
        tokens: list[str],
        n_size: int = 2,
        min_frequency: int = 2,
        top_k: int = 20,
    ) -> dict:
        if n_size < 2:
            n_size = 2

        ngrams = [
            tuple(tokens[index : index + n_size])
            for index in range(0, max(len(tokens) - n_size + 1, 0))
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
            pmi = math.log2((p_ngram / p_parts)) if p_parts > 0 else 0.0
            rows.append(
                {
                    "ngram_text": " ".join(ngram),
                    "ngram_size": n_size,
                    "frequency": freq,
                    "pmi_score": round(pmi, 4),
                }
            )
            if len(rows) >= top_k:
                break

        return {
            "analysis_type": "ngram_collocation",
            "ngram_size": n_size,
            "rows": rows,
        }
