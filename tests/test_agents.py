from agents.coordination_agent import CoordinationAgent
from agents.validation_agent import ValidationAgent


def test_frequency_query_routes_to_frequency_agent():
    coordinator = CoordinationAgent()
    payload = coordinator.execute(
        query="Show top 5 frequency words",
        tokens="cat dog cat bird cat".split(),
    )
    assert payload["safe"] is True
    assert payload["result"]["analysis_type"] == "frequency"
    assert payload["result"]["rows"][0]["word"] == "cat"


def test_kwic_query_returns_context_rows():
    coordinator = CoordinationAgent()
    payload = coordinator.execute(
        query="Generate KWIC for 'cat' with 2-word context",
        tokens="the cat sat on the mat cat jumps".split(),
    )
    assert payload["safe"] is True
    assert payload["result"]["analysis_type"] == "kwic"
    assert payload["result"]["matches"]


def test_keyword_comparison_requires_reference_corpus():
    coordinator = CoordinationAgent()
    payload = coordinator.execute(
        query="Compare keywords against reference corpus",
        tokens="cat cat dog".split(),
        reference_tokens=None,
    )
    assert payload["safe"] is False
    assert "reference corpus" in payload["validation"]["issues"][0].lower()


def test_dynamic_result_blocked_if_not_container_executed():
    validator = ValidationAgent()
    validation = validator.validate_result(
        {
            "analysis_type": "dynamic_code",
            "result": {"token_count": 10},
            "metadata": {"executed_in_container": False, "errors": []},
        }
    )
    assert validation["safe"] is False
    assert "container" in validation["issues"][0].lower()


# ---------------------------------------------------------------------------
# Query accuracy tests — deterministic output, exact correctness checks
# ---------------------------------------------------------------------------

def test_ngram_bigrams():
    """'the cat' appears twice in the input — must surface in top bigrams."""
    coordinator = CoordinationAgent()
    result = coordinator.execute(
        query="Find the top 3 bigrams",
        tokens="the cat sat on the mat the cat".split(),
    )
    assert result["safe"] is True
    # NgramAgent returns analysis_type "ngram_collocation"
    assert result["result"]["analysis_type"] == "ngram_collocation"
    ngrams = [r["ngram_text"] for r in result["result"]["rows"]]
    assert "the cat" in ngrams


def test_frequency_correct_counts():
    """Exact rank ordering: apple×3 > banana×2 > cherry×1."""
    coordinator = CoordinationAgent()
    tokens = "apple banana apple cherry apple banana".split()
    payload = coordinator.execute(
        query="Show top 10 frequency words",
        tokens=tokens,
    )
    assert payload["safe"] is True
    rows = payload["result"]["rows"]
    words_in_order = [r["word"] for r in rows]
    assert words_in_order[0] == "apple"
    assert words_in_order[1] == "banana"
    # Frequencies must be exact
    freq_map = {r["word"]: r["frequency"] for r in rows}
    assert freq_map["apple"] == 3
    assert freq_map["banana"] == 2
    assert freq_map["cherry"] == 1


def test_kwic_correct_context_windows():
    """KWIC for 'fox' should capture the correct left/right context."""
    coordinator = CoordinationAgent()
    tokens = "the quick brown fox jumps over the lazy dog".split()
    payload = coordinator.execute(
        query="Generate KWIC for 'fox' with 2-word context",
        tokens=tokens,
    )
    assert payload["safe"] is True
    matches = payload["result"]["matches"]
    assert len(matches) == 1
    match = matches[0]
    assert match["keyword"] == "fox"
    # Left context: "quick brown"  (window_size=2 by default or as extracted)
    assert "brown" in match["left_context"]
    assert "jumps" in match["right_context"]


def test_kwic_multiple_occurrences():
    """A keyword appearing three times should produce three concordance rows."""
    coordinator = CoordinationAgent()
    tokens = "cat sat cat on the mat cat".split()
    payload = coordinator.execute(
        query="Generate KWIC for 'cat'",
        tokens=tokens,
    )
    assert payload["safe"] is True
    assert payload["result"]["analysis_type"] == "kwic"
    assert len(payload["result"]["matches"]) == 3


def test_ngram_trigrams():
    """'the big cat' appears twice — must appear in trigram results."""
    coordinator = CoordinationAgent()
    tokens = "the big cat ate the big cat quickly".split()
    result = coordinator.execute(
        query="Find the top trigrams",
        tokens=tokens,
    )
    assert result["safe"] is True
    assert result["result"]["analysis_type"] == "ngram_collocation"
    ngrams = [r["ngram_text"] for r in result["result"]["rows"]]
    assert "the big cat" in ngrams
