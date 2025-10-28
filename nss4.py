import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer, util
from transformers import GenerationConfig


JSON_FILE = "nss_data.json"
BASE_MODEL = "distilgpt2"
MODEL_DIR = "./finetuned_nss_json_model"
EMB_PATH = "./embeddings.pt"


def flatten_nss_json(data):
    pairs = []

    
    about = data.get("about", {})
    for key, value in about.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                pairs.append((f"What is {subkey.replace('_',' ')}?", str(subval)))
        else:
            pairs.append((f"What is the {key} of NSS?", str(value)))

    
    for e in data.get("events", []):
        title = e.get("title", "")
        desc = e.get("description", "")
        date = e.get("date", "")
        pairs.append((f"What is {title}?", f"{desc} (Held on {date})"))

    for p in data.get("projects", []):
        title = p.get("title", "")
        desc = p.get("description", "")
        date = p.get("date", "")
        pairs.append((f"Tell me about project {title}.", f"{desc} (Date: {date})"))

    
    volunteer = data.get("volunteer", {})
    for key, value in volunteer.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                if isinstance(subval, list):
                    pairs.append((f"What are the {subkey} under {key}?", ", ".join(subval)))
                else:
                    pairs.append((f"What is {subkey} under {key}?", str(subval)))
        elif isinstance(value, list):
            pairs.append((f"What are {key}?", ", ".join(value)))
        else:
            pairs.append((f"What is {key}?", str(value)))

    for f in data.get("faq", []):
        q = f.get("question", "")
        a = f.get("answer", "")
        if q and a:
            pairs.append((q, a))

    return pairs

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

qa_pairs = flatten_nss_json(data)
texts = [f"Q: {q}\nA: {a}" for q, a in qa_pairs]
dataset = Dataset.from_dict({"text": texts})


if not os.path.exists(MODEL_DIR):
    print("No fine-tuned model found — starting fine-tuning on NSS JSON...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_data = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Fine-tuning complete and model saved.")
else:
    print("Fine-tuned model found — loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)


embedder = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(EMB_PATH):
    print(" Loading saved embeddings...")
    saved = torch.load(EMB_PATH)
    context_embeddings = saved["embeddings"]
    qa_pairs = saved["qa_pairs"]
else:
    print(" Generating embeddings for the first time...")
    context_embeddings = embedder.encode([a for _, a in qa_pairs], convert_to_tensor=True)
    torch.save({"embeddings": context_embeddings, "qa_pairs": qa_pairs}, EMB_PATH)
    print(" Embeddings generated and saved!")
def retrieve_relevant_text(question):
   
    q_emb = embedder.encode(question, convert_to_tensor=True)

    
    cosine_scores = util.cos_sim(q_emb, context_embeddings)[0]

    
    top_idx = torch.argmax(cosine_scores)

    
    answer = qa_pairs[top_idx][1]  
    score = cosine_scores[top_idx].item()
    return answer, score



DEFAULT_PROMPT = (
    "You are an intelligent and helpful assistant for DJS NSS (National Service Scheme). "
    "Answer ONLY using the information provided in the context. "
    "Do not make up facts or guess answers. "
    "If the answer is not available in the context, respond with 'Ask Yash!'. "
    "Keep your responses short, clear, and friendly (2–3 lines), so volunteers can quickly understand the information. "
    "Be polite, encouraging, and professional in tone."
)
def ask_bot(question, user_prompt=DEFAULT_PROMPT, max_new_tokens=80, retrieval_threshold=0.4):
   
    retrieved_text, r_score = retrieve_relevant_text(question)
    
   
    if r_score >= retrieval_threshold:
        return retrieved_text, r_score, r_score

    truncated_context = retrieved_text[:400]  

    prompt = (
        f"{user_prompt}\n\n"
        f"Context:\n{truncated_context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  
        repetition_penalty=1.8,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()

    
    confidence = (r_score - 0.3) / 0.7
    confidence = max(0.0, min(1.0, confidence))

    return answer, r_score, confidence


print("\n NSS Bot is ready! Type 'exit' to quit.\n")

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    answer, r_score, confidence = ask_bot(q)
    print(f"Bot: {answer}\nRetrieval Score: {r_score:.2f} | Confidence: {confidence:.2f}\n")