from uuid import uuid4
from typing import List
import io
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import requests

app = FastAPI()

OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://localhost:11434/api/embeddings")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")


def ensure_collection() -> None:
    resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if resp.status_code == 404:
        create_body = {"vectors": {"size": 768, "distance": "Cosine"}}
        r = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}", json=create_body
        )
        if r.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail="Failed to create collection")


def read_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def get_embedding(text: str) -> List[float]:
    resp = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={"model": OLLAMA_MODEL, "prompt": text},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Embedding service error")
    return resp.json().get("embedding", [])


def upload_to_qdrant(document_id: str, chunks: List[str]) -> None:
    ensure_collection()
    points = []
    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        points.append({
            "id": str(uuid4()),
            "vector": embedding,
            "payload": {"document_id": document_id, "text": chunk}
        })
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
        json={"points": points},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail="Failed to upload to Qdrant")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if file.content_type == "application/pdf":
        text = read_pdf(content)
    elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        text = read_docx(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    chunks = split_text(text)
    document_id = str(uuid4())
    upload_to_qdrant(document_id, chunks)
    return JSONResponse({"document_id": document_id, "segments_uploaded": len(chunks)})
