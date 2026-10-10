from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader

from chunking import chunk_text
from embedding_service import generate_embedding
from vector_store import store_embeddings, search_embeddings
from rag_service import generate_answer

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "AI Resume Assistant API is running"
    }


@app.post("/upload")
def upload_resume(file: UploadFile = File(...)):
    # Read PDF file
    reader = PdfReader(file.file)

    # Extract text from PDF
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Split resume text into chunks
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = []

    for chunk in chunks:
        vector = generate_embedding(chunk)
        embeddings.append(vector)

    # Store chunks and embeddings
    store_embeddings(chunks, embeddings)

    # Return upload information
    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "embedding": embeddings
    }


@app.post("/search")
def search_resume(question: str):

   # Convert question into an embedding
   query_embedding = generate_embedding(question)

   # Search ChromaDB
   results = search_embeddings(query_embedding)

   # Get relevant chunks
   relevant_chunks = results["documents"][0]

   # Combine chunks into one context
   context = "\n\n".join(relevant_chunks)

   # Generate AI answer
   answer = generate_answer(
       question,
       context
   )




   return {
       "question": question,
       "context": relevant_chunks,
       "answer": answer
   }
