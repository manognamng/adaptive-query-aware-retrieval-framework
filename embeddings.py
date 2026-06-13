from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(chunks):

    #embeddings = model.encode(chunks)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )
    
    return embeddings

def encode_query(query):

    #return model.encode([query])
    return model.encode(
        [query],
        normalize_embeddings=True
    )