from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings

def encode_query(query):

    return model.encode([query])