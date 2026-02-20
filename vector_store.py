import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        self.metadata = []

    def add(self, embeddings, chunks, metadata):
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            print("Vector index is empty; returning no results")
            return []
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx]
            })
        return results

    def save(self, path):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump((self.chunks, self.metadata), f)

    def load(self, path):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta.pkl", "rb") as f:
            self.chunks, self.metadata = pickle.load(f)

    def index_chunks(self, chunks):
        print(f"Indexing {len(chunks)} chunks")
        self.add(chunks)
