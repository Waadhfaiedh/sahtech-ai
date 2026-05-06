import os
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
gemini = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    language: str = "fr"  # "fr", "ar", or "en"

class AIResponse(BaseModel):
    should_ai_answer: bool
    answer: str | None = None

SYSTEM_PROMPT = """You are a helpful medical assistant for Sahtech, a healthcare platform in Tunisia.
You answer simple, general health questions based ONLY on the provided context documents.
Rules:
- If the question is simple and the context contains enough information, answer it clearly and kindly.
- If the question requires a doctor's personal judgment, diagnosis, or is too complex, respond with exactly: "NEEDS_SPECIALIST"
- Never invent medical information not found in the context.
- Reply in the same language as the patient's message ({language}).
- Keep answers short and reassuring.
"""

@app.post("/ask", response_model=AIResponse)
async def ask(req: MessageRequest):
    # Search for relevant documents
    results = db.similarity_search_with_score(req.message, k=3)
    
    # If best match score is too low (not relevant), let specialist handle it
    if not results or results[0][1] > 1.2:  # ChromaDB distance, higher = less similar
        return AIResponse(should_ai_answer=False)
    
    context = "\n\n".join([doc.page_content for doc, score in results])
    
    prompt = f"""Context from medical documents:
{context}

Patient question: {req.message}

{SYSTEM_PROMPT.format(language=req.language)}"""

    response = gemini.generate_content(prompt)
    answer = response.text.strip()
    
    if answer == "NEEDS_SPECIALIST" or "NEEDS_SPECIALIST" in answer:
        return AIResponse(should_ai_answer=False)
    
    return AIResponse(should_ai_answer=True, answer=answer)

@app.get("/health")
def health():
    return {"status": "ok"}