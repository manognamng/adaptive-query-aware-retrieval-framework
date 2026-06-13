#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_TYPES = ["sentence", "paragraph", "section"]

#TOP_K = 5
TOP_K = 15

DATA_PATH = "data/sample_research_paper.pdf"

VECTOR_DB_PATH = "vector_store/"