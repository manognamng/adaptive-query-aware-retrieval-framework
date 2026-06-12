from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query, retrieved_chunks):

    pairs = [
        (query, item["text"])
        for item in retrieved_chunks
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(retrieved_chunks, scores):
        item["rerank_score"] = float(score)

    ranked = sorted(
        retrieved_chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked