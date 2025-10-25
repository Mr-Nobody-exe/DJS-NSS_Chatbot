from modules.rag_pipeline import NSSRAGPipeline

if __name__ == "__main__":
    rag = NSSRAGPipeline()
    print("NSS RAG Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag.run(query)
        print(f"NSSBot: {answer}\n")
