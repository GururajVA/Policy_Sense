from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from decision_llm import initialize_llm
from document_extractor import extract_text
from chunk_and_embed import chunk_text, embed_chunks
from vector_store import VectorStore
from query_parser import parse_query
from semantic_search import retrieve_clauses
from decision_llm import make_decision

import os

app = FastAPI()

# Enable CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Health RAG API running"}


@app.on_event("startup")
async def warmup_model():
    # Preload LLM to reduce first-request latency
    ok = initialize_llm()
    print(f"LLM warmup: {'ok' if ok else 'failed'}")
vector_store = None
embedding_dim = 384  # For MiniLM-L6-v2

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    file_path = f"uploaded_docs/{file.filename}"
    os.makedirs("uploaded_docs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)
    text = extract_text(file_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    metadata = [{"filename": file.filename, "chunk_id": i} for i in range(len(chunks))]
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(embedding_dim)
    vector_store.add(embeddings, chunks, metadata)
    vector_store.save("vector_store/index")
    return {"status": "uploaded and indexed", "chunks": len(chunks)}

@app.post("/query/")
async def query_policy(request: Request, query: str | None = Form(default=None)):
    """Accepts either form data (field 'query') or JSON body {"query": "..."}."""
    try:
        if query is None:
            # Attempt to read JSON if form field not provided
            try:
                payload = await request.json()
                query = payload.get("query")
            except Exception:
                query = None

        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="Missing 'query' in form data or JSON body")

        print(f"Received query: {query}")

        global vector_store
        if vector_store is None:
            vector_store = VectorStore(embedding_dim)
            try:
                vector_store.load("vector_store/index")
            except Exception as load_err:
                # If no index yet, continue with empty store
                print(f"Vector store load failed or empty: {load_err}")

        # Parse query with graceful degradation
        try:
            parsed = parse_query(query)
        except Exception as parse_err:
            print(f"parse_query failed: {parse_err}")
            parsed = {"age": None, "gender": None, "location": None, "procedure": None, "duration": None}
        print(f"Parsed query: {parsed}")

        # Retrieve relevant chunks safely
        try:
            retrieved = retrieve_clauses(vector_store, query)
        except Exception as retr_err:
            print(f"retrieve_clauses failed: {retr_err}")
            retrieved = []
        print(f"Retrieved chunks: {len(retrieved)}")

        # Generate decision
        try:
            decision = make_decision(query, parsed, retrieved)
        except Exception as llm_err:
            print(f"make_decision failed: {llm_err}")
            # Fallback minimal response when LLM unavailable
            decision = json.dumps({
                "decision": "unknown",
                "amount": 0,
                "justification": f"LLM unavailable. Retrieved clauses considered: {len(retrieved)}"
            })

        print(f"Generated response: {decision}")
        return {"result": decision}
    except HTTPException:
        raise
    except Exception as e:
        print(f"/query unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")