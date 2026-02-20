from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_embedding(query):
    return model.encode([query])[0]

def retrieve_clauses(vector_store, query, top_k=5):
    query_embedding = get_query_embedding(query)
    return vector_store.search(query_embedding, top_k=top_k)

def semantic_search(parsed_query):
    results = vector_store.search(parsed_query)
    print(f"Semantic search results: {results}")
    return results