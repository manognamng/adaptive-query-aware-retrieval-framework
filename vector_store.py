import faiss
import numpy as np

class VectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatL2(dimension)

        self.documents = []

        self.metadata = []
        
    def add_embeddings(
        self,
        embeddings,
        chunks,
        chunk_type
    ):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(embeddings)

        self.documents.extend(chunks)

        for i, chunk in enumerate(chunks):

            self.metadata.append({
                "chunk_id": i,
                "chunk_type": chunk_type,
                "text_length": len(chunk)
            })

    def search(self, query_embedding, top_k=5):

        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            chunk = self.documents[idx]

            if len(chunk.split()) < 8:
                continue

            results.append({
                "text": chunk,
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results