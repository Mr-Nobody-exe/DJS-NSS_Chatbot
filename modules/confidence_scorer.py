from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import OPENAI_API_KEY, EMBEDDING_MODEL

class ConfidenceScorer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    def compute_confidence(self, context: str, response: str) -> float:
        """
        Calculates cosine similarity between context and model response.
        Returns a float between 0 (unrelated) and 1 (identical).
        """
        if not context.strip() or not response.strip():
            return 0.0
        
        context_vec = self.embeddings.embed_query(context)
        response_vec = self.embeddings.embed_query(response)
        similarity = cosine_similarity([context_vec], [response_vec])[0][0]
        return round(float(similarity), 3)
