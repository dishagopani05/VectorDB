# Vector Database API

This FastAPI-based API provides endpoints to manage and search vectorized documents using **ChromaDB, FAISS, and Pinecone**.

## Features
- **ChromaDB**: Store and retrieve documents with embeddings.
- **FAISS**: Efficient similarity search.
- **Pinecone**: Scalable vector search.
- **FastAPI**: High-performance API framework.

## Installation

```bash
git clone https://github.com/your-username/vector-database-api.git
cd vector-database-api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload

API Endpoints

----------------------------------------------------------------------
ChromaDB

POST /chroma/add – Add a document
GET /chroma/documents – Get all documents
POST /chroma/search – Search for similar documents
PUT /chroma/update – Update a document
DELETE /chroma/delete/{doc_id} – Delete a document
DELETE /chroma/delete_collection – Delete all documents

-------------------------------------------------------------------------
FAISS

POST /faiss/add_vector – Add a document
GET /faiss/get_vector/{doc_id} – Retrieve a vector
GET /faiss/search?query=<text>&top_k=3 – Search for similar vectors
DELETE /faiss/delete_vector/{doc_id} – Delete a vector

-------------------------------------------------------------------------
Pinecone

POST /pinecone/add – Add a document
PUT /pinecone/update_vector/{index_id} – Update a vector
GET /pinecone/get_vector/{doc_id} – Retrieve a vector
DELETE /pinecone/delete_vector/{index_id}/{doc_id} – Delete a vector