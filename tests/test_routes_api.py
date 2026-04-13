import io

from app import create_app


def _upload_text(client, filename: str, body: str):
    data = {"file": (io.BytesIO(body.encode("utf-8")), filename)}
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


def test_upload_and_frequency_query_flow():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = _upload_text(client, "freq_sample.txt", "cat dog cat bird")
    assert response.status_code == 200
    corpus_id = response.get_json()["corpus_id"]

    ask_response = client.post(
        "/api/query",
        json={"question": "show frequency top 5", "corpus_id": corpus_id},
    )
    assert ask_response.status_code == 200
    payload = ask_response.get_json()
    assert payload["safe"] is True
    assert payload["result"]["analysis_type"] == "frequency"


def test_keyword_query_with_reference_corpus():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    _upload_text(client, "target_corpus.txt", "cat cat dog fish")
    _upload_text(client, "reference_corpus.txt", "dog dog fish fish")

    response = client.post(
        "/api/query",
        json={
            "question": "Compare keywords against reference corpus",
            "corpus_id": "target_corpus.txt",
            "reference_corpus_id": "reference_corpus.txt",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result"]["analysis_type"] == "keyword_comparison"


def test_out_of_scope_prompt_is_rejected():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _upload_text(client, "scope_sample.txt", "one two three")

    response = client.post(
        "/api/query",
        json={"question": "What is the weather tomorrow?", "corpus_id": "scope_sample.txt"},
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["safe"] is False
    assert "outside the scope" in payload["error"].lower()
