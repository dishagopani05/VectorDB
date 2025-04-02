from fastapi import FastAPI

from routers.chroma_router import router as chroma_router 
from routers.pinecone_router import router as pinecone_router
from routers.faiss_router import router as faiss_router

app = FastAPI(title="Vector Database API", description="API for various vector databases")

# Including different vector database routers
app.include_router(chroma_router, prefix="/chroma", tags=["ChromaDB"])
app.include_router(pinecone_router, prefix="/pinecone", tags=["Pinecone"])
app.include_router(faiss_router, prefix="/faiss", tags=["FAISS"])

@app.get("/")
def root():
    return {"message": "Welcome to the Vector Database API"}

