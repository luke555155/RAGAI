import logging
from typing import List, Dict

import requests
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class QdrantClient:
    """簡易 Qdrant API 包裝"""

    def __init__(self, url: str, collection: str):
        self.url = url.rstrip('/')
        self.collection = collection

    def ensure_collection(self) -> None:
        """確認指定的 collection 存在，不存在則嘗試建立"""
        try:
            resp = requests.get(f"{self.url}/collections/{self.collection}")
            if resp.status_code == 404:
                create_body = {"vectors": {"size": 768, "distance": "Cosine"}}
                r = requests.put(
                    f"{self.url}/collections/{self.collection}", json=create_body
                )
                if r.status_code not in (200, 201):
                    raise HTTPException(status_code=500, detail="Failed to create collection")
        except Exception as exc:
            logger.exception("Ensure collection error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc

    def upload_points(self, points: List[Dict]) -> None:
        """將多個 points 上傳至 Qdrant"""
        try:
            self.ensure_collection()
            resp = requests.put(
                f"{self.url}/collections/{self.collection}/points?wait=true",
                json={"points": points},
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail="Failed to upload to Qdrant")
        except Exception as exc:
            logger.exception("Upload points error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc

    def search(self, vector: List[float], limit: int = 3) -> List[Dict]:
        """根據向量進行相似度搜尋"""
        try:
            resp = requests.post(
                f"{self.url}/collections/{self.collection}/points/search",
                json={"vector": vector, "limit": limit},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Qdrant search error")
            return resp.json().get("result", [])
        except Exception as exc:
            logger.exception("Search error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc
