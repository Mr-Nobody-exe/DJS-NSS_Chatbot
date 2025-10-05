from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # type: ignore

MODEL_ID = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,           
    max_length=256,      # longer max length for fuller responses
    do_sample=True,      # sampling instead of greedy decoding
    top_p=0.9,           # nucleus sampling for natural diversity
    top_k=50,            
    temperature=0.7     
)

def generate_response(prompt: str) -> str:
    instruction = f"You are a helpful assistant. Answer clearly and concisely:\n{prompt}"
    output = generator(instruction, max_length=256, do_sample=True, top_p=0.9, top_k=50, temperature=0.7)
    response = output[0].get("generated_text", "").strip()
    
    if response.endswith((" .", " ,", " ;")):
        response = response[:-2] + response[-1]
    
    return response
