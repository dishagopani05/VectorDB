from fastapi import APIRouter
import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

router = APIRouter()

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="documents")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class Document(BaseModel):
    id: str
    content: str

class QueryRequest(BaseModel):
    query: str
    n_results: int = 1

@router.post("/add")
def add_document(doc: Document):
    collection.add(ids=[doc.id], documents=[doc.content])
    return {"message": "Document added", "id": doc.id}

@router.get("/documents")
def get_all_documents():
    return collection.get()

@router.post("/search")
def search_documents(query: QueryRequest):
    query_vector = embedding_model.encode([query.query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=query.n_results)
    return results

@router.put("/update")
def update_document(doc: Document):
    collection.delete(ids=[doc.id])
    collection.add(ids=[doc.id], documents=[doc.content])
    return {"message": "Document updated", "id": doc.id}

@router.delete("/delete/{doc_id}")
def delete_document(doc_id: str):
    collection.delete(ids=[doc_id])
    return {"message": "Document deleted", "id": doc_id}

@router.delete("/delete_collection")
def delete_collection():
    chroma_client.delete_collection(name="documents")
    return {"message": "Collection deleted"}
