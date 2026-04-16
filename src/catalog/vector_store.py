"""
FAISS-backed vector store for cosine similarity search over market embeddings.

Uses IndexFlatIP (inner product) on L2-normalised vectors, which is equivalent
to cosine similarity but SIMD-accelerated — ~20x faster than raw numpy at this scale.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = "data/vectors"


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []

    def add(self, embeddings: list[list[float]], metadata: list[dict]) -> None:
        """
        Build the index from a list of embedding vectors and their associated
        metadata dicts (one per vector, same order).
        """
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = metadata
        logger.info(
            "VectorStore built",
            extra={"vectors": len(embeddings), "dim": vectors.shape[1]},
        )

    def search(self, query_embedding: list[float], k: int = 30) -> list[dict]:
        """
        Return the top-k most similar documents to the query vector.
        Each result is the metadata dict with an added "score" key (cosine similarity).
        """
        if self.index is None:
            raise RuntimeError("VectorStore is empty — call add() or load() first")
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.documents[idx], "score": float(score)})
        return results

    def save(self, path: str = _DEFAULT_PATH) -> None:
        """
        Persist the FAISS index to {path}.faiss and metadata to {path}_meta.json.
        """
        if self.index is None:
            raise RuntimeError("Nothing to save — VectorStore is empty")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        Path(f"{path}_meta.json").write_text(json.dumps(self.documents))
        logger.info(
            "VectorStore saved",
            extra={"path": path, "count": len(self.documents)},
        )

    def load(self, path: str = _DEFAULT_PATH) -> None:
        """
        Load a previously saved index and its metadata from disk.
        """
        self.index = faiss.read_index(f"{path}.faiss")
        self.documents = json.loads(Path(f"{path}_meta.json").read_text())
        logger.info(
            "VectorStore loaded",
            extra={"path": path, "count": len(self.documents)},
        )
