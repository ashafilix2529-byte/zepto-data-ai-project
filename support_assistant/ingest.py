from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
COLLECTION = "zepto_policies"


def main():
    client = chromadb.PersistentClient(path=str(DB))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer("all-MiniLM-L6-v2")

    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8")
        ids.append(path.stem)
        documents.append(text)
        metadatas.append({"document_id": path.stem})

    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} policy documents in ChromaDB.")


if __name__ == "__main__":
    main()
