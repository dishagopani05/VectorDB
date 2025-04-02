import faiss
import numpy as np
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from typing import List

router = APIRouter()

encoder = SentenceTransformer("all-MiniLM-L6-v2") 

vector_dim = 384 
index = faiss.IndexFlatL2(vector_dim) 
# faiss.normalize_L2(index.reconstruct_n(0, index.ntotal)) 

doc_id_to_index = {}

@router.post("/add_vector/")
def add_vector(doc_id: str, content: str):
    """Add a new text vector to FAISS"""
    vector = encoder.encode([content])
    vector = np.array(vector, dtype=np.float32)
    faiss.normalize_L2(vector)

    idx = index.ntotal  # Get the next available FAISS index
    index.add(vector)  # Add vector to FAISS
    doc_id_to_index[doc_id] = idx  # Map document ID to FAISS index

    return {"message": "Vector added", "doc_id": doc_id}

@router.get("/get_vector/{doc_id}")
def get_vector(doc_id: str):
    """Retrieve vector by document ID"""
    if doc_id not in doc_id_to_index:
        raise HTTPException(status_code=404, detail="Vector not found")

    idx = doc_id_to_index[doc_id]
    vector = index.reconstruct(idx)  # Retrieve vector

    return {"doc_id": doc_id, "vector": vector.tolist()}

@router.get("/search/")
def search(query: str, top_k: int = 3):
    """Search for the most similar vectors"""
    query_vector = encoder.encode([query])
    query_vector = np.array(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    distances, ann = index.search(query_vector, k=top_k)  # Perform FAISS search

    results = [
        {"doc_id": doc_id, "distance": float(dist)}
        for doc_id, dist in zip(doc_id_to_index.keys(), distances[0])
    ]

    return {"query": query, "results": results}

@router.delete("/delete_vector/{doc_id}")
def delete_vector(doc_id: str):
    """Delete a vector by document ID"""
    if doc_id not in doc_id_to_index:
        raise HTTPException(status_code=404, detail="Vector not found")

    del doc_id_to_index[doc_id]  # Remove from mapping
    return {"message": "Vector deleted", "doc_id": doc_id}
