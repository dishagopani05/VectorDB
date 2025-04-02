import os
from fastapi import APIRouter
import pinecone
from pinecone import Pinecone, ServerlessSpec
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

router = APIRouter()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key="pcsk_4SKbBF_SRL1nL1QRxU55RwuLvYjkJTWuj9kM324tYwUYQg5KVU9fhRQqfUpraHwquvR4Pp")
INDEX_NAME = "doc"  
print(pc.list_indexes().names())
try:
     
    if INDEX_NAME not in pc.list_indexes().names():

        pc.create_index(
            name=INDEX_NAME,
            dimension=384, 
            metric="cosine",  
            spec=ServerlessSpec(cloud="aws", region="us-east-1") 
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    index = pc.Index(INDEX_NAME)
    print(f"Connected to index: {INDEX_NAME}")

except Exception as e:
    print(f"Error: {e}")
    
class Document(BaseModel):
    id: str
    content: str
    metadata: dict = {}  
    

@router.post("/add")
def add_document(doc: Document):
    # Convert content to vector
    if not doc.content:
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")
    
    vector = embedding_model.encode(doc.content).tolist()

    # Upsert document to Pinecone
    response = index.upsert(vectors=[{"id": doc.id, "values": vector, "metadata": {"content": doc.content}}])
    print(f"Pinecone Upsert Response: {response}")
    
    return {"message": "Document added", "id": doc.id}

@router.put("/update_vector/{index_id}")
def update_vector(index_id: str, doc: Document):
    try:
        if not doc.content:
            raise HTTPException(status_code=400, detail="Document content cannot be empty.")
        index_exists = index.describe_index_stats().get('namespaces', {}).get(index_id)
        if not index_exists:
            raise HTTPException(status_code=404, detail=f"Index with id {index_id} does not exist.")
        vector = embedding_model.encode(doc.content).tolist()
    
        response = index.upsert(vectors=[{"id": doc.id, "values": vector, "metadata": {"content": doc.content}}])
        print(f"Pinecone Upsert Response: {response}")
    
        return {"message": "Document added", "id": doc.id}

    except pinecone.exceptions.PineconeException as e:
        raise HTTPException(status_code=500, detail=f"Error upserting vector: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.get("/get_vector/{doc_id}")
def get_vector(doc_id: str):
    try:
        response = index.fetch(ids=[doc_id]) 
        print(response)
        if not hasattr(response, "vectors") or doc_id not in response.vectors:
            raise HTTPException(status_code=404, detail="Vector not found.")

        vector_data = response.vectors[doc_id] 

        return {
            "doc_id": doc_id,
            "vector": vector_data.values,  
            "metadata": vector_data.metadata  
        }

    except pinecone.exceptions.PineconeException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching vector: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



@router.delete("/delete_vector/{index_id}/{doc_id}")
def delete_vector(index_id: str, doc_id: str):
    try:
        # Delete the vector from Pinecone by doc_id
        response = index.delete(ids=[doc_id])

        # Check if the delete operation was successful
        if response.get("deleted", 0) == 0:
            raise HTTPException(status_code=404, detail="Vector not found for deletion.")

        # Return success message
        return {"message": f"Vector '{doc_id}' deleted from index '{index_id}'."}

    except pinecone.exceptions.PineconeException as e:
        # Handle Pinecone-specific errors
        raise HTTPException(status_code=500, detail=f"Error deleting vector: {str(e)}")
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")