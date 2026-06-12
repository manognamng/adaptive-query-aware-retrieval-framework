# answer_generator.py

def generate_answer(query, results):

    top_chunks = [
        r["text"]
        for r in results[:3]
    ]

    answer = " ".join(top_chunks)

    return answer