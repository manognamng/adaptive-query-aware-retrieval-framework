import re
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize


# Clean text
def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Sliding window sentence chunks
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


# Paragraph chunks
def paragraph_chunks(text):

    paragraphs = text.split("\\n\\n")

    cleaned = [
        clean_text(p)
        for p in paragraphs
        if len(p.strip()) > 50
    ]

    return cleaned


# Section chunks
def section_chunks(
    text,
    section_size=4
):

    paragraphs = paragraph_chunks(text)

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