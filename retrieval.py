from embeddings import encode_query

from config import TOP_K

def retrieve(query, vector_stores, predictor):

    granularity = predictor(query)

    query_embedding = encode_query(query)

    store = vector_stores[granularity]

    results = store.search(
        query_embedding,
        TOP_K
    )

    return granularity, results