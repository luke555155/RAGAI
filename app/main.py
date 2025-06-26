from uuid import uuid4
from typing import List
import io
import re
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import requests

from .services.qdrant_client import QdrantClient

app = FastAPI()

OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama2"

qdrant_client = QdrantClient(QDRANT_URL, COLLECTION_NAME)




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
    """Split text into chunks. Handles Chinese by sentence punctuation."""
    if re.search(r"[\u4e00-\u9fff]", text):
        sentences = re.split(r"(?<=[。！？])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 500:
                chunks.append(current)
                current = sent
            else:
                current += sent
        if current:
            chunks.append(current)
        return chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


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
        raise HTTPException(status_code=500, detail=f"Embedding service error: {exc}")


def upload_to_qdrant(document_id: str, chunks: List[str], source_file: str) -> None:
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
                    "source_file": source_file,
                    "upload_time": upload_time,
                },
            }
        )
    qdrant_client.upload_points(points)




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
    upload_to_qdrant(document_id, chunks, file.filename)
    return JSONResponse({"document_id": document_id, "segments_uploaded": len(chunks)})


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask(req: AskRequest):
    q_embedding = get_embedding(req.question)
    results = qdrant_client.search(q_embedding, limit=3)
    context = "\n".join(r["payload"].get("text", "") for r in results)
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt},
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ollama service error: {exc}")
    return {"answer": answer, "references": [r["payload"] for r in results]}
