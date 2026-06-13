from embeddings import encode_query
from config import TOP_K
from reranker import rerank


def retrieve(query, vector_stores, predictor):

    granularity = predictor(query)

    query_embedding = encode_query(query)

    store = vector_stores[granularity]

    print(f"\nSearching Top {TOP_K} chunks...")

    results = store.search(
        query_embedding,
        TOP_K
    )

    results = rerank(
        query,
        results
    )

    return granularity, results