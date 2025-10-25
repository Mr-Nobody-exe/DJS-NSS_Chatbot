from modules.confidence_scorer import ConfidenceScorer

def validate_response(query: str, context: str, response: str) -> str:
    scorer = ConfidenceScorer()
    confidence = scorer.compute_confidence(context, response)

    if context.strip() == "" or response.strip() == "":
        return "I'm not sure about that. Please check the official NSS notice board or coordinator."

    if confidence < 0.6:
        return (
            f"I couldn’t verify that answer confidently (confidence={confidence}). "
            "Please refer to official NSS records for confirmation."
        )

    if any(word in response.lower() for word in ["guess", "maybe", "probably"]):
        return "I couldn't confirm that from NSS data. Please refer to verified sources."

    return response
