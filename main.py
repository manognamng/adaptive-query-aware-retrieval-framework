import os
import faiss
import pickle
from document_loader import load_pdf_text
from answer_generator import generate_answer

from chunking import (
    sliding_window_chunks,
    paragraph_chunks,
    section_chunks
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

section_data = section_chunks(text)


# TEMP LIMITS FOR FASTER TESTING
sentence_data = sentence_data[:80]

paragraph_data = paragraph_data[:40]

section_data = section_data[:15]


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

print("\nRetrieval Statistics")
print("-" * 30)
print(f"Granularity Used: {granularity}")
print(f"Chunks Retrieved: {len(results)}")

answer = generate_answer(
    query,
    results
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
print("GENERATED ANSWER")
print("="*60)

print(answer)

print("\n" + "="*60)
print("RETRIEVED EVIDENCE")
print("="*60)


print("\nRetrieval Statistics")
print("-" * 30)
print(f"Granularity Used: {granularity}")
print(f"Chunks Retrieved: {len(results)}")

for idx, result in enumerate(results, 1):

    print(f"\nResult {idx}")

    print("Score:", result["score"])

    print("Chunk Type:",
          result["metadata"]["chunk_type"])

    print("\nText:")

    print(result["text"])