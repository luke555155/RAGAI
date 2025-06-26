import requests
from fastapi import HTTPException
from typing import List, Dict


class QdrantClient:
    def __init__(self, url: str, collection: str):
        self.url = url.rstrip('/')
        self.collection = collection

    def ensure_collection(self) -> None:
        resp = requests.get(f"{self.url}/collections/{self.collection}")
        if resp.status_code == 404:
            create_body = {"vectors": {"size": 768, "distance": "Cosine"}}
            r = requests.put(
                f"{self.url}/collections/{self.collection}", json=create_body
            )
            if r.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail="Failed to create collection")

    def upload_points(self, points: List[Dict]) -> None:
        self.ensure_collection()
        resp = requests.put(
            f"{self.url}/collections/{self.collection}/points?wait=true",
            json={"points": points},
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail="Failed to upload to Qdrant")

    def search(self, vector: List[float], limit: int = 3) -> List[Dict]:
        resp = requests.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json={"vector": vector, "limit": limit},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Qdrant search error")
        return resp.json().get("result", [])
