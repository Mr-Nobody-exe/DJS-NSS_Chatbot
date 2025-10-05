import json
from langchain.docstore.document import Document  # type: ignore
from modules.retriever import build_retriever
from modules.prompt_engineering import create_prompt
from modules.llm_wrapper import generate_response

def flatten_nss_json_for_documents(data):
    docs = []
    # About section
    if 'about' in data:
        for k, v in data['about'].items():
            if isinstance(v, dict):
                text = f"{k}: " + ', '.join([f"{kk}: {vv}" for kk, vv in v.items()])
                docs.append(Document(page_content=text))
            else:
                docs.append(Document(page_content=f"{k}: {v}"))
    # Events section
    if 'events' in data:
        for event in data['events']:
            info = '\n'.join([f"{ek}: {ev}" for ek, ev in event.items()])
            docs.append(Document(page_content=info))
    # Projects section
    if 'projects' in data:
        for project in data['projects']:
            info = '\n'.join([f"{pk}: {pv}" for pk, pv in project.items()])
            docs.append(Document(page_content=info))
    # Volunteer section
    if 'volunteer' in data:
        vol = data['volunteer']
        # Overview
        if 'overview' in vol:
            docs.append(Document(page_content=vol['overview']['title'] + ": " + vol['overview']['description']))
        # Impact
        if 'impact' in vol:
            imp = vol['impact']
            docs.append(Document(page_content=imp['title'] + ": " + imp['description']))
            if 'benefits_summary' in imp:
                docs.append(Document(page_content="Benefits Summary: " + imp['benefits_summary']))
        # Benefits and lists
        if 'benefits' in vol:
            docs.append(Document(page_content="Benefits: " + '; '.join(vol['benefits'])))
        for key in ['why_volunteer', 'how_to_get_involved']:
            if key in vol:
                docs.append(Document(page_content=f"{key.replace('_',' ').title()}: " + '; '.join(vol[key])))
        # Policy
        if 'policy' in vol:
            for pk, pv in vol['policy'].items():
                if isinstance(pv, list):
                    docs.append(Document(page_content=f"{pk.replace('_',' ').title()}: " + '; '.join(pv)))
                else:
                    docs.append(Document(page_content=f"{pk.replace('_',' ').title()}: {pv}"))
        # Impact Statement and application
        if 'impact_statement' in vol:
            docs.append(Document(page_content="Impact Statement: " + vol['impact_statement']))
        if 'application_process' in vol:
            docs.append(Document(page_content="Application Process: " + '; '.join(vol['application_process'])))
    # FAQ section
    if 'faq' in data:
        for faq in data['faq']:
            docs.append(Document(page_content=f"Q: {faq['question']} A: {faq['answer']}"))
    return docs

class NSSChatbot:
    def __init__(self):
        with open("modules/nss_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.docs = flatten_nss_json_for_documents(data)
        self.retriever = build_retriever(self.docs)

    def ask(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "Sorry, I couldn't find any information on that."
        context = "\n".join([doc.page_content for doc in docs])
        prompt = create_prompt(context, query)
        response = generate_response(prompt)
        return response
