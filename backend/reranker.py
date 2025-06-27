import json
import logging
import os
from datetime import datetime
from typing import List, Dict
import re

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "llama2")
OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama"""
    resp = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={"model": OLLAMA_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    emb = resp.json().get("embedding")
    if emb is None:
        raise ValueError("No embedding returned")
    return emb


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_passages_by_cosine(passages: List[Dict]) -> List[Dict]:
    """Sort passages by similarity score descending"""
    ranked = sorted(passages, key=lambda r: r.get("score", 0), reverse=True)
    for idx, p in enumerate(ranked, 1):
        p["rank"] = idx
    return ranked


def rerank_passages_with_llm(question: str, passages: List[Dict], log_path: str | None = "logs/rerank_log.jsonl") -> List[Dict]:
    """Use LLM to rank passages"""
    prompt_lines = [
        "請根據問題對以下段落按相關度由高到低排序，僅回傳段落編號，以逗號分隔。",
        f"問題: {question}",
    ]
    for i, p in enumerate(passages, 1):
        text = p.get("payload", {}).get("text", "")
        prompt_lines.append(f"({i}) {text}")
    prompt = "\n".join(prompt_lines)
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": OLLAMA_LLM_MODEL, "prompt": prompt},
    )
    resp.raise_for_status()
    answer = resp.json().get("response", "")
    numbers = [int(n) for n in re.findall(r"\d+", answer)]
    order = [n - 1 for n in numbers if 0 < n <= len(passages)]
    ranked = [passages[i] for i in order]
    others = [p for i, p in enumerate(passages) if i not in order]
    ranked.extend(others)
    for idx, p in enumerate(ranked, 1):
        p["rank"] = idx
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        before = [p.get("id") for p in passages]
        after = [p.get("id") for p in ranked]
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question,
                    "before": before,
                    "after": after,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    return ranked



def rerank_passages(question: str, passages: List[Dict], mode: str = "llm") -> List[Dict]:
    """Entry point for passage reranking"""
    if mode == "cosine":
        return rerank_passages_by_cosine(passages)
    return rerank_passages_with_llm(question, passages)

