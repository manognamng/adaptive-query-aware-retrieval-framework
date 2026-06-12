def generate_explanation(query, granularity):

    if granularity == "sentence":

        return (
            "Fine-grained retrieval selected "
            "because the query is factual."
        )

    elif granularity == "paragraph":

        return (
            "Paragraph retrieval selected "
            "because moderate context is required."
        )

    elif granularity == "section":

        return (
            "Section-level retrieval selected "
            "because long contextual reasoning is needed."
        )

    return "Default retrieval strategy selected."