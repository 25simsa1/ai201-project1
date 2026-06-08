"""
Milestone 4 - embedding and retrieval.

Takes the chunks from ingest.py (chunks.json), embeds them with
all-MiniLM-L6-v2 and stores them in a local ChromaDB collection. Also has the
search() function the rest of the project uses to pull the top-k chunks for a
query (top-k = 5, from planning.md).
"""

import json

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "colby_reviews"
EMBED_MODEL = "all-MiniLM-L6-v2"   # from planning.md
TOP_K = 5                          # from planning.md

# Let Chroma embed both the chunks and the queries with the same MiniLM model,
# so a stored chunk and an incoming question land in the same vector space.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def build_store():
    """Load chunks.json and (re)build the Chroma collection from scratch."""
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # wipe any old collection so re-running doesn't pile up duplicates
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete the first time
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[str(c["chunk_id"]) for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "professor": c["professor"]}
            for c in chunks
        ],
    )
    print(f"Embedded and stored {collection.count()} chunks in '{COLLECTION_NAME}'")
    return collection


def search(query, k=TOP_K):
    """Return the top-k chunks most similar to the query."""
    collection = get_collection()
    res = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "professor": meta["professor"],
        })
    return hits


if __name__ == "__main__":
    build_store()

    # quick check: run a sample query and see if the retrieved chunks make sense
    sample = "Which professors are engaging in lectures?"
    print(f"\nSample query: {sample}\n")
    for i, hit in enumerate(search(sample), 1):
        print(f"{i}. {hit['professor']} ({hit['source']})")
        print(f"   {hit['text'][:160].strip()}...\n")
