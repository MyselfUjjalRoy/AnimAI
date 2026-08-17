"""
RAG Retriever — searches Manim docs for relevant patterns
given a user query.

Gracefully returns empty results if the index is not built or
dependencies (faiss / sentence-transformers) are not installed.
"""

import json
import os

# Load once globally so we don't reload every request
_model = None
_index = None
_chunks = None
_load_failed = False


def load_rag():
    """Load the FAISS index and chunks into memory."""
    global _model, _index, _chunks, _load_failed

    if _model is not None or _load_failed:
        return  # already loaded or permanently failed

    index_path = os.path.join(os.path.dirname(__file__), "manim_docs.index")
    chunks_path = os.path.join(os.path.dirname(__file__), "manim_chunks.json")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[RAG] Index files not found — skipping RAG (run rag/download_docs.py to build)")
        _load_failed = True
        return

    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        print("[RAG] Loading search index...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')

        with open(chunks_path, "r") as f:
            _chunks = json.load(f)

        _index = faiss.read_index(index_path)
        print(f"[RAG] Loaded {len(_chunks)} chunks ready to search")

    except ImportError as e:
        print(f"[RAG] Missing dependency ({e}) — RAG disabled. Install faiss-cpu and sentence-transformers.")
        _load_failed = True
    except Exception as e:
        print(f"[RAG] Failed to load index: {e} — RAG disabled")
        _load_failed = True


def retrieve(query: str, k: int = 4) -> str:
    """
    Searches the Manim docs index for chunks relevant to the query.
    Returns a formatted string of the top-k results to inject into prompt.
    Returns empty string if RAG is not available.
    """
    try:
        load_rag()
    except Exception:
        return ""

    if _model is None or _index is None or _chunks is None:
        return ""

    try:
        import numpy as np

        # Embed the query
        query_embedding = _model.encode([query], convert_to_numpy=True).astype(np.float32)

        # Search FAISS
        distances, indices = _index.search(query_embedding, k)

        # Format results for prompt injection
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_chunks):
                chunk = _chunks[idx]
                results.append(f"# Relevant Manim pattern {i+1} (from {chunk['source']}):\n{chunk['content']}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""


if __name__ == "__main__":
    # Quick test
    result = retrieve("animate binary search tree nodes")
    print("Search results:")
    print(result[:500])