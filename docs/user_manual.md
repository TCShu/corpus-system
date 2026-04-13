# ACAS User Manual

## 1. Start the System

1. Activate virtual environment:
   - `.\venv\Scripts\activate`
2. Run:
   - `python run.py`
3. Open:
   - `http://127.0.0.1:5000`

## 2. Upload a Corpus

1. Click **Upload Corpus**.
2. Select a file (`.txt`, `.csv`, `.json`, `.xml`).
3. Wait for upload confirmation.
4. The corpus appears in the **Projects** list and corpus selector.

## 3. Ask for Analysis

Type a natural-language query in the input area, then click **Analyze**.

Examples:
- `Show top 20 frequency words excluding stopwords`
- `Generate KWIC for "language" with 10-word context`
- `Find bigram collocations`
- `Compare keywords against reference corpus`

For keyword comparison, select a **Reference Corpus**.

## 4. Safety and Validation

- Every result is validated by the WatchDog agent.
- If the query is out of ACAS scope, it is rejected.
- If the query is linguistic but not supported by existing agents, ACAS attempts restricted dynamic execution in Docker.
- If Docker execution is unavailable or unsafe, ACAS blocks the result.

## 5. Result Interpretation

Each response includes:
- `safe`: whether the result passed validation
- `validation`: issues/warnings
- `result`: analysis payload

## 6. API Quick Reference

- `POST /api/upload`: upload corpus file
- `GET /api/corpora`: list uploaded corpora
- `POST /api/query`: run orchestrated linguistic analysis
- `GET /health`: service health check

Legacy:
- `POST /upload`
- `POST /ask`
