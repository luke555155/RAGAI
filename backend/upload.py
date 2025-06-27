import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import HTTPException

from .services.qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME = "documents"
OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "llama2")

qdrant_client = QdrantClient(QDRANT_URL, COLLECTION_NAME)

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_INDEX_FILE = BASE_DIR / "doc_index.json"

def load_doc_index() -> dict:
    if DOC_INDEX_FILE.exists():
        try:
            with open(DOC_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to load doc index: %s", exc)
    return {}

def save_doc_index(index: dict) -> None:
    with open(DOC_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def append_doc_entry(document_id: str, summary: str, tags: list[str]) -> None:
    """Add or update document entry with summary and tags"""
    index = load_doc_index()
    index[document_id] = {"summary": summary, "tags": tags}
    save_doc_index(index)

def remove_doc_entry(document_id: str) -> None:
    """Remove document entry from index"""
    index = load_doc_index()
    if document_id in index:
        index.pop(document_id, None)
        save_doc_index(index)

def get_embedding(text: str) -> List[float]:
    try:
        resp = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={"model": OLLAMA_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
        if embedding is None:
            raise ValueError("No embedding returned")
        return embedding
    except Exception as exc:
        logger.exception("Embedding service error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding service error: {exc}") from exc

def upload_document(document_id: str, chunks: List[str], file_name: str, tags: list[str]) -> str:
    points = []
    upload_time = datetime.utcnow().isoformat()
    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        points.append(
            {
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "text": chunk,
                    "chunk_index": idx,
                    "file_name": file_name,
                    "upload_time": upload_time,
                },
            }
        )
    qdrant_client.upload_points(points)
    context = "\n".join(chunks)
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_LLM_MODEL, "prompt": f"請根據以下內容提供一段摘要：\n{context}"},
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "")
    except Exception as exc:
        logger.exception("Ollama summary error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ollama service error: {exc}") from exc
    append_doc_entry(document_id, summary, tags)
    return summary
