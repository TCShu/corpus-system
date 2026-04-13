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
