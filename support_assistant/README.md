# Support Assistant

## Architecture

`docs/*.txt` → `ingest.py` chunk/document loading → SentenceTransformer `all-MiniLM-L6-v2` embeddings → ChromaDB collection `zepto_policies` → LangGraph `classify_intent` → conditional routing → `retrieve_and_answer` or `direct_answer` → Pydantic `Answer` → FastAPI `/ask`.

The supplied documents are used as the corpus. Given their short length, each document is stored as one chunk.

## Mock mode

`MOCK_LLM` defaults to enabled. The classifier uses the specified keyword heuristic. Policy questions perform real embedding + ChromaDB retrieval, while answer generation is deterministic. General questions return the required fixed canned response. No LLM API call is needed.

## Structured prompt

The `structured_prompt()` function contains all required skeleton components:
role, context, task, format, length, plus an explicit negative constraint and a few-shot example. It is intended for the optional real-LLM generation path.

## API examples

After `python ingest.py` and starting uvicorn:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{"query":"What is the delivery fee below INR 149?"}"

curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{"query":"Tell me a joke"}"
```

Record the actual JSON responses produced on your machine in the final README before submission.

## Docker

Build and run locally:

```bash
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```
