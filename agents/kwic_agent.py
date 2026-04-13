class KWICAgent:
    def analyze(
        self,
        tokens: list[str],
        keyword: str,
        window_size: int = 5,
        max_results: int = 50,
    ) -> dict:
        keyword = keyword.lower().strip()
        matches = []
        for index, token in enumerate(tokens):
            if token != keyword:
                continue
            left = " ".join(tokens[max(0, index - window_size) : index])
            right = " ".join(tokens[index + 1 : index + 1 + window_size])
            matches.append(
                {
                    "position": index,
                    "left_context": left,
                    "keyword": keyword,
                    "right_context": right,
                }
            )
            if len(matches) >= max_results:
                break

        return {
            "analysis_type": "kwic",
            "keyword": keyword,
            "window_size": window_size,
            "matches": matches,
        }
