from langchain_huggingface import HuggingFaceEndpoint
import os

def generate_response(prompt: str) -> str:
    # Use HuggingFaceEndpoint instead of HuggingFaceHub
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",   # or tiiuae/falcon-7b-instruct
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    return llm.invoke(prompt)
