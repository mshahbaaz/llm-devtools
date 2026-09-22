from __future__ import annotations

import chromadb
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    document: str
    distance: float


class VectorStore:
    """In-memory Chroma collection using its default embedding function.

    Swappable embedding models happen at the caller level by constructing
    multiple VectorStore instances with different `embedding_function` values
    (Task 3.3 wires this into config).
    """

    def __init__(self, collection_name: str, embedding_function=None):
        self._client = chromadb.EphemeralClient()
        kwargs = {"name": collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**kwargs)

    def add(self, ids: list[str], documents: list[str]) -> None:
        self._collection.add(ids=ids, documents=documents)

    def query(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(query_texts=[query_text], n_results=top_k)
        chunks = []
        for id_, doc, dist in zip(result["ids"][0], result["documents"][0], result["distances"][0]):
            chunks.append(RetrievedChunk(id=id_, document=doc, distance=dist))
        return chunks
