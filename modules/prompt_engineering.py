from langchain.prompts import PromptTemplate # type: ignore
def create_prompt(context: str, query: str) -> str:
    prompt = f"""
You are a helpful, polite assistant. Based on the information below, please:

- Provide a clear and concise answer.
- Rephrase or combine related facts smoothly.
- Break down long or complex information into easy-to-read sentences.
- Use polite conversational cues and transitions.
- If the information is incomplete or not available, say so politely.
- Encourage the user to ask follow-up questions if needed.

Information:
{context}

Question:
{query}

Answer:
"""
    return prompt
