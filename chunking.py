import re
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------------
# SENTENCE CHUNKS
# -------------------------------
def sliding_window_chunks(
    text,
    window_size=3,
    overlap=1
):

    text = clean_text(text)

    sentences = sent_tokenize(text)

    chunks = []

    step = window_size - overlap

    for i in range(0, len(sentences), step):

        chunk = " ".join(
            sentences[i:i + window_size]
        )

        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks


# -------------------------------
# PARAGRAPH CHUNKS
# -------------------------------
def paragraph_chunks(
    text,
    chunk_size=5
):

    text = clean_text(text)

    sentences = sent_tokenize(text)

    chunks = []

    for i in range(
        0,
        len(sentences),
        chunk_size
    ):

        chunk = " ".join(
            sentences[i:i + chunk_size]
        )

        # Skip very short chunks
        if len(chunk.split()) < 10:
            continue

        # Skip chunks starting with citations
        if chunk.strip().startswith("["):
            continue

        # Skip DOI links
        if "doi.org" in chunk.lower():
            continue

        # Skip References section
        if "references" in chunk.lower():
            continue

        # Skip citation-heavy chunks
        citation_count = chunk.count("[")

        if citation_count > 3:
            continue

        chunks.append(chunk)

    print(
        "Filtered paragraph chunks:",
        len(chunks)
    )

    return chunks


# -------------------------------
# SECTION CHUNKS
# -------------------------------
def section_chunks(
    text,
    section_size=1
):

    paragraphs = paragraph_chunks(text)

    print(
        "Paragraph chunks:",
        len(paragraphs)
    )

    if len(paragraphs) > 0:

        print(
            "First chunk length:",
            len(paragraphs[0])
        )

    sections = []

    for i in range(
        0,
        len(paragraphs),
        section_size
    ):

        section = " ".join(
            paragraphs[i:i + section_size]
        )

        sections.append(section)

    return sections