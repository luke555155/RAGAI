from uuid import uuid4
from typing import List
from time import time
import io
import re
import json
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import requests
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

from .services.qdrant_client import QdrantClient
from .upload import (
    upload_document,
    load_doc_index,
    save_doc_index,
    remove_doc_entry,
)
from .reranker import rerank_passages, cosine_similarity

app = FastAPI()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME = "documents"
OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "llama2")
RERANK_MODE = os.getenv("RERANK_MODE", "llm")
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "8"))

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))
CACHE_TTL = int(os.getenv("ASK_CACHE_TTL", "300"))
ASK_CACHE: dict = {}

qdrant_client = QdrantClient(QDRANT_URL, COLLECTION_NAME)




def read_pdf(file_bytes: bytes) -> str:
    """讀取 PDF 檔案並回傳文字內容"""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx(file_bytes: bytes) -> str:
    """讀取 Word 檔案並回傳文字內容"""
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def split_text(text: str) -> List[str]:
    """將文本依長度切分為數個區塊，中文以句號等標點為界"""
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
    """向 Ollama 取得文字向量，失敗時回傳 500"""
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


def count_tokens(text: str) -> int:
    """Rudimentary token count based on whitespace splitting"""
    return len(text.split())






@app.post("/api/upload")
async def upload(file: UploadFile = File(...), tags: str = Form("")):
    """上傳檔案並分割後寫入 Qdrant"""
    content = await file.read()
    if file.content_type == "application/pdf":
        text = read_pdf(content)
    elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        text = read_docx(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    chunks = split_text(text)
    document_id = str(uuid4())
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    summary = upload_document(document_id, chunks, file.filename, tag_list)
    return JSONResponse({"document_id": document_id, "segments_uploaded": len(chunks), "summary": summary})


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None
    document_ids: List[str] | None = None
    style: str | None = None
    prompt_template: str | None = None
    rerank_mode: str | None = None
    include_filtered: bool | None = True


class DeleteDocRequest(BaseModel):
    document_id: str


class RankTestRequest(BaseModel):
    question: str
    passages: List[str]
    mode: str | None = None


class ManualAskRequest(BaseModel):
    question: str
    segments: List[str]
    style: str | None = None
    prompt_template: str | None = None


@app.get("/api/docs")
async def list_docs():
    """回傳所有文件資訊"""
    try:
        docs = qdrant_client.list_documents()
        summaries = load_doc_index()
        for doc in docs:
            doc_id = doc.get("document_id")
            if doc_id in summaries:
                entry = summaries[doc_id]
                if isinstance(entry, dict):
                    doc["summary"] = entry.get("summary")
                    doc["tags"] = entry.get("tags", [])
                else:
                    # backward compatibility if index contains only summary
                    doc["summary"] = entry
                    doc["tags"] = []
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("List docs error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"documents": docs}


@app.get("/api/docs/{document_id}")
async def get_document(document_id: str):
    """取得指定文件的段落內容"""
    try:
        segments = qdrant_client.get_segments_by_doc(document_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Get document error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"document_id": document_id, "segments": segments}


@app.delete("/api/docs")
async def delete_document(req: DeleteDocRequest):
    """刪除指定文件的所有資料"""
    try:
        qdrant_client.delete_by_document(req.document_id)
        remove_doc_entry(req.document_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Delete document error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "deleted"}


@app.post("/api/docs/{document_id}/resummarize")
async def resummarize_document(document_id: str):
    """重新取得段落並產生摘要"""
    try:
        segments = qdrant_client.get_segments_by_doc(document_id)
        if not segments:
            raise HTTPException(status_code=404, detail="Document not found")
        context = "\n".join(s.get("text", "") for s in segments)
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_LLM_MODEL, "prompt": f"請根據以下內容提供一段摘要：\n{context}"},
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "")
        index = load_doc_index()
        entry = index.get(document_id, {})
        entry["summary"] = summary
        if "tags" not in entry:
            entry["tags"] = []
        index[document_id] = entry
        save_doc_index(index)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Resummarize error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"document_id": document_id, "summary": summary}


@app.post("/api/rank_test")
async def rank_test(req: RankTestRequest):
    """Return ranking results for provided passages"""
    try:
        q_emb = get_embedding(req.question)
        passages = []
        for idx, text in enumerate(req.passages):
            emb = get_embedding(text)
            score = cosine_similarity(q_emb, emb)
            passages.append({"id": str(idx), "payload": {"text": text}, "vector": emb, "score": score})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Rank test error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    ranked = rerank_passages(req.question, passages, mode=req.mode or RERANK_MODE)
    refs = [
        {"text": p["payload"].get("text", ""), "rank": p.get("rank"), "score": p.get("score"), "filtered": p.get("filtered")}
        for p in ranked
    ]
    return {"answer": "", "references": refs}




@app.post("/api/ask")
async def ask(req: AskRequest):
    """根據問題向 Qdrant 取得相關段落並呼叫 LLM 回答"""
    cache_key = (
        req.question,
        tuple(req.document_ids or ([req.document_id] if req.document_id else [])),
        req.style,
        req.prompt_template,
        req.rerank_mode,
        req.include_filtered,
    )
    cached = ASK_CACHE.get(cache_key)
    if cached and time() - cached["ts"] < CACHE_TTL:
        return cached["data"]

    start_time = time()
    q_embedding = get_embedding(req.question)
    try:
        results = qdrant_client.search(
            q_embedding,
            limit=SEARCH_LIMIT,
            document_id=req.document_id,
            document_ids=req.document_ids,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search Qdrant error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {exc}") from exc

    ranked_all = rerank_passages(req.question, results, mode=req.rerank_mode or RERANK_MODE)
    for idx, r in enumerate(ranked_all, 1):
        if (req.rerank_mode or RERANK_MODE) == "llm" and idx > 5:
            r["filtered"] = True
    context_parts = []
    used_results = []
    tokens = 0
    for r in ranked_all:
        if r.get("filtered") and not req.include_filtered:
            continue
        text = r["payload"].get("text", "")
        t = count_tokens(text)
        if tokens + t > MAX_CONTEXT_TOKENS:
            break
        tokens += t
        context_parts.append(text)
        used_results.append(r)
    context = "\n".join(context_parts)

    template = (
        req.prompt_template
        or "Answer the question based on the context below.\n{style}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = template.format(
        context=context,
        question=req.question,
        style=req.style or "",
    )
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt},
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "")
    except Exception as exc:
        logger.exception("Ollama service error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ollama service error: {exc}") from exc
    references = [
        {
            "text": r["payload"].get("text", ""),
            "chunk_index": r["payload"].get("chunk_index"),
            "score": r.get("score"),
            "rank": r.get("rank"),
            "filtered": r.get("filtered"),
        }
        for r in used_results
    ]
    elapsed = time() - start_time
    data = {"answer": answer, "references": references, "elapsed": elapsed}
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/ask_log.jsonl", "a", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": req.question,
                    "document_ids": req.document_ids
                    or ([req.document_id] if req.document_id else []),
                    "style": req.style,
                    "rerank_mode": req.rerank_mode or RERANK_MODE,
                    "answer": answer,
                    "references": references,
                    "elapsed": elapsed,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    except Exception as exc:
        logger.warning("Failed to log ask: %s", exc)
    ASK_CACHE[cache_key] = {"ts": time(), "data": data}
    return data


@app.get("/api/ask_log")
async def get_ask_log():
    """Return logged ask records"""
    try:
        entries = []
        if os.path.exists("logs/ask_log.jsonl"):
            with open("logs/ask_log.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return {"logs": entries[::-1]}
    except Exception as exc:
        logger.exception("Read ask log error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read log") from exc


@app.post("/api/manual-ask")
async def manual_ask(req: ManualAskRequest):
    """Answer a question using manually provided context"""
    start_time = time()
    context = "\n".join(req.segments)
    template = (
        req.prompt_template
        or "Answer the question based on the context below.\n{style}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = template.format(context=context, question=req.question, style=req.style or "")
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt},
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "")
    except Exception as exc:
        logger.exception("Ollama service error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ollama service error: {exc}") from exc
    elapsed = time() - start_time
    refs = [{"text": s} for s in req.segments]
    return {"answer": answer, "references": refs, "elapsed": elapsed}
