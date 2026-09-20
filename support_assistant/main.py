import os
from pathlib import Path
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent
DB = ROOT / "chroma_db"
COLLECTION = "zepto_policies"
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

KEYWORDS = ["delivery","return","refund","membership","tracking","cancel","gift card","support hours"]


class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)


class AskRequest(BaseModel):
    query: str


class State(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list
    response: Answer


client = chromadb.PersistentClient(path=str(DB))
collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def structured_prompt(context, query):
    return f"""ROLE: You are a Zepto policy support assistant.
CONTEXT: Answer only from the retrieved Zepto policy context below.
TASK: Answer the customer's question using only that context.
FORMAT: Return JSON with answer, sources, and confidence.
LENGTH: Keep the answer concise, no more than 80 words.
NEGATIVE CONSTRAINT: Do not answer using information not present in the provided context.
FEW-SHOT EXAMPLE:
Question: What is the delivery fee below INR 149?
Answer: Standard delivery below INR 149 costs INR 25.
RETRIEVED CONTEXT:
{context}
CUSTOMER QUESTION:
{query}
"""


def classify_intent(state):
    q = state["query"].lower()
    intent = "policy_question" if any(k in q for k in KEYWORDS) else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state):
    query = state["query"]
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=3)
    docs = result["documents"][0]
    ids = result["ids"][0]
    top = docs[0]
    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top[:200]}"
        return {"retrieved": list(zip(ids, docs)),
                "response": Answer(answer=answer, sources=ids, confidence=1.0)}
    # Optional real-LLM extension point. Keep generation grounded in structured_prompt.
    prompt = structured_prompt("\n\n".join(docs), query)
    # Replace this block with the selected free-tier LLM client when MOCK_LLM=0.
    raise RuntimeError("MOCK_LLM=0 requires configuring a real LLM backend.")


def direct_answer(state):
    if MOCK_LLM:
        return {"response": Answer(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[], confidence=1.0)}
    raise RuntimeError("MOCK_LLM=0 requires configuring a real LLM backend.")


def route(state):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)
graph.set_entry_point("classify_intent")
graph.add_conditional_edges("classify_intent", route, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer"
})
graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)
app_graph = graph.compile()

app = FastAPI(title="Zepto Policy Support Assistant")


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest):
    result = app_graph.invoke({"query": req.query})
    return result["response"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
