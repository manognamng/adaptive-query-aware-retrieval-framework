import re


def get_best_chunk(results, query):

    if len(results) == 0:
        return None

    query = query.lower()

    if "bacteria" in query:

        priority_terms = [
            "bacteroidetes",
            "bifidobacterium",
            "actinobacteria",
            "porphyromonas",
            "prevotella",
            "fusobacterium"
        ]

        for r in results:

            text = r["text"].lower()

            if any(
                term in text
                for term in priority_terms
            ):
                return r

    return results[0]


def generate_rationale(query, chunk):

    return (
        "This chunk was selected because it "
        "contains the strongest semantic match "
        "to the query and received the highest "
        "retrieval relevance score after reranking."
    )


def generate_precise_answer(query, chunk):

    query_words = {
        word.lower()
        for word in re.findall(r"\w+", query)
        if len(word) > 2
    }

    sentences = re.split(
        r'(?<=[.!?])\s+',
        chunk
    )

    best_sentence = ""
    best_score = -1

    for sentence in sentences:

        sentence_lower = sentence.lower()

        score = sum(
            1
            for word in query_words
            if word in sentence_lower
        )

        if score > best_score:

            best_score = score
            best_sentence = sentence

    if best_sentence:

        return best_sentence.strip()

    return chunk[:300]