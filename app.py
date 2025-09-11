# app.py
from fastapi import FastAPI, Query
from modules.orchestrator import NSSChatbot

# Initialize chatbot once at startup
chatbot = NSSChatbot()
app = FastAPI()

@app.get("/chat")
def chat(query: str = Query(..., description="Your question to the chatbot")):
    return {"answer": chatbot.ask(query)}

# Testing 
if __name__ == "__main__":
    print("Chatbot console mode (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("Bot:", chatbot.ask(query), "\n")
