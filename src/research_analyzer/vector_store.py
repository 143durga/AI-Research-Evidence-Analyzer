"""Persistent Chroma storage using its built-in local ONNX text embeddings."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from research_analyzer.pdf_processor import TextChunk


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    distance: float


class PaperVectorStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.client = chromadb.PersistentClient(path=str(storage_path))

    @staticmethod
    def document_id(pdf_bytes: bytes) -> str:
        import hashlib

        return hashlib.sha256(pdf_bytes).hexdigest()[:24]

    def index_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
    ) -> None:
        collection = self.client.get_or_create_collection(
            name=f"paper_{document_id}", metadata={"hnsw:space": "cosine"}
        )
        ids = [
            f"{document_id}_p{chunk.page_number:04d}_c{chunk.chunk_index:04d}"
            for chunk in chunks
        ]
        collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"filename": filename, "page_number": chunk.page_number}
                for chunk in chunks
            ],
        )

    def search(
        self, document_id: str, question: str, top_k: int = 5
    ) -> list[Evidence]:
        collection = self.client.get_collection(name=f"paper_{document_id}")
        if collection.count() == 0:
            return []

        result = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        ids = result["ids"][0]
        return [
            Evidence(
                chunk_id=chunk_id,
                text=text,
                filename=metadata["filename"],
                page_number=int(metadata["page_number"]),
                distance=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]