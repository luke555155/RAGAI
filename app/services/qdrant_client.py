import logging
from typing import List, Dict, Optional

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

    def search(
        self,
        vector: List[float],
        limit: int = 3,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """根據向量進行相似度搜尋，可依 document_id 或多個 document_ids 篩選"""
        try:
            body: Dict = {"vector": vector, "limit": limit}
            if document_ids:
                body["filter"] = {
                    "must": [
                        {"key": "document_id", "match": {"any": document_ids}}
                    ]
                }
            elif document_id:
                body["filter"] = {
                    "must": [{"key": "document_id", "match": {"value": document_id}}]
                }
            resp = requests.post(
                f"{self.url}/collections/{self.collection}/points/search",
                json=body,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Qdrant search error")
            return resp.json().get("result", [])
        except Exception as exc:
            logger.exception("Search error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc

    def list_documents(self) -> List[Dict]:
        """列出所有已上傳的文件"""
        try:
            self.ensure_collection()
            docs: Dict[str, Dict[str, str]] = {}
            body: Dict[str, Optional[Dict]] = {
                "limit": 100,
                "with_payload": ["document_id", "file_name", "upload_time"],
                "with_vector": False,
            }
            offset = None
            while True:
                if offset:
                    body["offset"] = offset
                resp = requests.post(
                    f"{self.url}/collections/{self.collection}/points/scroll",
                    json=body,
                    timeout=30,
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to scroll points")
                data = resp.json()
                for point in data.get("result", data.get("points", [])):
                    payload = point.get("payload", {})
                    doc_id = payload.get("document_id")
                    if doc_id and doc_id not in docs:
                        docs[doc_id] = {
                            "document_id": doc_id,
                            "file_name": payload.get("file_name"),
                            "upload_time": payload.get("upload_time"),
                        }
                offset = data.get("next_page_offset") or data.get("next_offset")
                if not offset:
                    break
            return list(docs.values())
        except Exception as exc:
            logger.exception("List documents error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc

    def get_segments_by_doc(self, document_id: str) -> List[Dict]:
        """取得指定文件的所有段落"""
        try:
            self.ensure_collection()
            body = {
                "limit": 100,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [{"key": "document_id", "match": {"value": document_id}}]
                },
            }
            offset = None
            segments: List[Dict] = []
            while True:
                if offset:
                    body["offset"] = offset
                resp = requests.post(
                    f"{self.url}/collections/{self.collection}/points/scroll",
                    json=body,
                    timeout=30,
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to scroll points")
                data = resp.json()
                for point in data.get("result", data.get("points", [])):
                    segments.append(point.get("payload", {}))
                offset = data.get("next_page_offset") or data.get("next_offset")
                if not offset:
                    break
            segments.sort(key=lambda x: x.get("chunk_index", 0))
            return segments
        except Exception as exc:
            logger.exception("Get segments error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc

    def delete_by_document(self, document_id: str) -> None:
        """刪除指定文件的所有向量"""
        try:
            self.ensure_collection()
            body = {
                "filter": {
                    "must": [{"key": "document_id", "match": {"value": document_id}}]
                }
            }
            resp = requests.post(
                f"{self.url}/collections/{self.collection}/points/delete?wait=true",
                json=body,
                timeout=30,
            )
            if resp.status_code not in (200, 202):
                raise HTTPException(status_code=500, detail="Failed to delete from Qdrant")
        except Exception as exc:
            logger.exception("Delete document error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {exc}") from exc
