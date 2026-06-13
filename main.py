import os
import faiss
import pickle
from document_loader import load_pdf_text
from answer_generator import generate_answer
from reranker import rerank

from chunking import (
    sliding_window_chunks,
    paragraph_chunks,
    section_chunks
)

from answer_formatter import (
    get_best_chunk,
    generate_rationale,
    generate_precise_answer
)

from embeddings import generate_embeddings

from vector_store import VectorStore

from granularity_predictor import predict_granularity

from retrieval import retrieve

from explainability import generate_explanation


PDF_PATH = "data/alzheimers_microbiota_paper.pdf"


# -------------------------------
# LOAD PDF
# -------------------------------

print("Loading PDF...")

text = load_pdf_text(PDF_PATH)


# -------------------------------
# CHUNK GENERATION
# -------------------------------

print("Generating chunks...")

sentence_data = sliding_window_chunks(text)

paragraph_data = paragraph_chunks(text)

print(
    "Paragraph chunks:",
    len(paragraph_data)
)

if len(paragraph_data) > 0:

    print(
        "First chunk length (chars):",
        len(paragraph_data[0])
    )

    print(
        "First chunk length (words):",
        len(paragraph_data[0].split())
    )
    
section_data = section_chunks(text)


# TEMP LIMITS FOR FASTER TESTING
#sentence_data = sentence_data[:80]

#paragraph_data = paragraph_data[:40]

#section_data = section_data[:15]


# -------------------------------
# EMBEDDING GENERATION
# -------------------------------

print("Generating sentence embeddings...")

sentence_embeddings = generate_embeddings(
    sentence_data
)

print("Generating paragraph embeddings...")

paragraph_embeddings = generate_embeddings(
    paragraph_data
)

print("Generating section embeddings...")

section_embeddings = generate_embeddings(
    section_data
)


# -------------------------------
# VECTOR DATABASES
# -------------------------------

print("Building vector stores...")

sentence_store = VectorStore(
    sentence_embeddings.shape[1]
)

paragraph_store = VectorStore(
    paragraph_embeddings.shape[1]
)

section_store = VectorStore(
    section_embeddings.shape[1]
)


# -------------------------------
# ADD EMBEDDINGS
# -------------------------------

sentence_store.add_embeddings(
    sentence_embeddings,
    sentence_data,
    "sentence"
)

paragraph_store.add_embeddings(
    paragraph_embeddings,
    paragraph_data,
    "paragraph"
)

section_store.add_embeddings(
    section_embeddings,
    section_data,
    "section"
)

faiss.write_index(
    sentence_store.index,
    "vector_store/sentence.index"
)

with open(
    "vector_store/sentence_metadata.pkl",
    "wb"
) as f:
    pickle.dump(sentence_store.metadata, f)

with open(
    "vector_store/paragraph_metadata.pkl",
    "wb"
) as f:
    pickle.dump(paragraph_store.metadata, f)

with open(
    "vector_store/section_metadata.pkl",
    "wb"
) as f:
    pickle.dump(section_store.metadata, f)


faiss.write_index(
    paragraph_store.index,
    "vector_store/paragraph.index"
)

faiss.write_index(
    section_store.index,
    "vector_store/section.index"
)

with open("vector_store/sentence_metadata.pkl", "wb") as f:
    pickle.dump(sentence_store.metadata, f)

with open("vector_store/paragraph_metadata.pkl", "wb") as f:
    pickle.dump(paragraph_store.metadata, f)

with open("vector_store/section_metadata.pkl", "wb") as f:
    pickle.dump(section_store.metadata, f)
    
# -------------------------------
# STORE ALL VECTOR DBS
# -------------------------------

vector_stores = {
    "sentence": sentence_store,
    "paragraph": paragraph_store,
    "section": section_store
}


# -------------------------------
# USER QUERY
# -------------------------------

query = input("Enter your query: ")


# -------------------------------
# RETRIEVAL
# -------------------------------

granularity, results = retrieve(
    query,
    vector_stores,
    predict_granularity
)

results = sorted(
    results,
    key=lambda x: x["similarity"],
    reverse=True
)

#results = rerank(
#    query,
#    results
#)

print("\nTOP RETRIEVED CHUNKS")
print("-" * 50)

for i, r in enumerate(results[:5], 1):

    print(f"\nChunk {i}")

    print(
        "Similarity:",
        round(r["similarity"], 4)
    )

    print(r["text"][:500])

    print("-" * 50)
    
print("\nRetrieval Statistics")
print("-" * 30)
print(f"Granularity Used: {granularity}")
print(f"Chunks Retrieved: {len(results)}")

#best_chunk = get_best_chunk(results)
print("\nTOP 3 RETRIEVED CANDIDATES")
print("="*60)

for i, r in enumerate(results[:3], 1):

    print(f"\nCandidate {i}")

    print("Similarity:",
          round(r["similarity"], 4))

    print(r["text"][:200])
    

print("\nDEBUG BEST CHUNK SELECTION")

for i, r in enumerate(results[:5], 1):

    print(
        i,
        round(r["similarity"], 4)
    )

    print(r["text"][:150])
    
#best_result = get_best_chunk(results)
get_best_chunk(results, query)
if "bacteria" in query.lower():

    best_chunk = " ".join(
        [
            r["text"]
            for r in results[:5]
        ]
    )

else:

    best_chunk = best_result["text"]
    

import re


    
print("\nDEBUG")
print("Best chunk length:", len(best_chunk))
print("Word count:", len(best_chunk.split()))

#best_result = get_best_chunk(results)
best_result = get_best_chunk(
    results,
    query
)

print(
    "Semantic Similarity:",
    round(best_result["similarity"], 4)
)

#print(
#    "CrossEncoder Score:",
#    round(best_result["rerank_score"], 4)
#)

best_chunk = best_result["text"]

rationale = generate_rationale(
    query,
    best_chunk
)

answer = generate_precise_answer(
    query,
    best_chunk
)

print(
    f"Retrieved because query requires "
    f"{granularity}-level contextual information."
)

# -------------------------------
# EXPLAINABILITY
# -------------------------------

print("\nPredicted Granularity:", granularity)

print("\n" + "="*60)
print("CHUNKED DATA")
print("="*60)

sentences = best_chunk.split(".")

best_chunk = max(
    sentences,
    key=lambda s: sum(
        word.lower() in s.lower()
        for word in query.split()
    )
)
print(best_chunk.strip() + ".")
print(best_chunk)

print("\n" + "="*60)
print("RATIONALE")
print("="*60)

print(rationale)

print("\n" + "="*60)
print("ACTUAL ANSWER")
print("="*60)

print(answer)
print("\nCONFIDENCE")
print("=" * 60)

print(
    "Semantic Similarity:",
    round(best_result["similarity"], 4)
)

#print(
#    "CrossEncoder Score:",
#    round(best_result["rerank_score"], 4)
#)

print("\nRetrieval Statistics")
print("-" * 30)
print(f"Granularity Used: {granularity}")
print(f"Chunks Retrieved: {len(results)}")
print("Retrieval Method: FAISS + CrossEncoder Reranking")

print(
    "Similarity:",
    round(r["similarity"], 4)
)

#print(
#    "Rerank:",
#    round(r["rerank_score"], 4)
#)