from collections import Counter


class KeywordAgent:
    def analyze(
        self,
        target_tokens: list[str],
        reference_tokens: list[str],
        top_k: int = 20,
        min_frequency: int = 2,
    ) -> dict:
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
            rows.append(
                {
                    "word": word,
                    "target_frequency": target_freq,
                    "reference_frequency": ref_counts.get(word, 0),
                    "keyness_ratio": round(keyness, 6),
                }
            )

        rows.sort(key=lambda item: item["keyness_ratio"], reverse=True)
        return {
            "analysis_type": "keyword_comparison",
            "rows": rows[:top_k],
        }
