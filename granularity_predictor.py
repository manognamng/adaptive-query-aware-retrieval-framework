def predict_granularity(query):

    query = query.lower()

    bacteria_terms = [
        "bacteria",
        "microbiota",
        "microbiome",
        "species"
    ]

    mechanism_terms = [
        "mechanism",
        "pathway",
        "role",
        "explain"
    ]

    therapy_terms = [
        "therapy",
        "treatment",
        "probiotic",
        "supplementation"
    ]

    if any(t in query for t in bacteria_terms):
        return "paragraph"

    elif any(t in query for t in mechanism_terms):
        return "section"

    elif any(t in query for t in therapy_terms):
        return "paragraph"


    return "paragraph"