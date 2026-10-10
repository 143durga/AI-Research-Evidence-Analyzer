"""Persistent Chroma storage using its built-in local ONNX text embeddings."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from research_analyzer.definitions import abbreviation_definitions, requested_acronym
from research_analyzer.pdf_processor import TextChunk


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    distance: float | None
    direct_match: bool = False


class PaperVectorStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=Settings(anonymized_telemetry=False),
        )

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
        existing_ids = collection.get(include=[])["ids"]
        stale_ids = set(existing_ids) - set(ids)
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
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
        collection_name = f"paper_{document_id}"
        collection_names = {
            getattr(collection, "name", collection)
            for collection in self.client.list_collections()
        }
        if collection_name not in collection_names:
            return []
        collection = self.client.get_collection(name=collection_name)
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
        evidence = [
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
        acronym = requested_acronym(question)
        if not acronym:
            return evidence

        existing_ids = {item.chunk_id for item in evidence}
        all_chunks = collection.get(include=["documents", "metadatas"])
        for chunk_id, text, metadata in zip(
            all_chunks["ids"],
            all_chunks["documents"],
            all_chunks["metadatas"],
            strict=True,
        ):
            if not abbreviation_definitions(text, acronym):
                continue
            if chunk_id in existing_ids:
                evidence = [
                    Evidence(
                        item.chunk_id,
                        item.text,
                        item.filename,
                        item.page_number,
                        item.distance,
                        direct_match=True,
                    )
                    if item.chunk_id == chunk_id
                    else item
                    for item in evidence
                ]
                continue
            evidence.append(
                Evidence(
                    chunk_id=chunk_id,
                    text=text,
                    filename=metadata["filename"],
                    page_number=int(metadata["page_number"]),
                    distance=None,
                    direct_match=True,
                )
            )
        return evidence