import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = Ollama(model="llama3.2", base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    language: str = "fr"

class AIResponse(BaseModel):
    should_ai_answer: bool
    answer: str | None = None

SYSTEM_PROMPT = """You are a helpful medical assistant for Sahtech, a healthcare platform in Tunisia.
Answer simple health questions based ONLY on the provided context.
Rules:
- If simple and context is sufficient, answer clearly and kindly.
- If it requires a doctor's personal judgment or diagnosis, reply with exactly: NEEDS_SPECIALIST
- Never invent information not in the context.
- Reply in the same language as the patient's message.
- Keep answers short.
"""

@app.post("/ask", response_model=AIResponse)
async def ask(req: MessageRequest):
    results = db.similarity_search_with_score(req.message, k=3)

    if not results or results[0][1] > 1.8:
        return AIResponse(should_ai_answer=False)

    context = "\n\n".join([doc.page_content for doc, score in results])

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Patient question: {req.message}
Answer:"""

    answer = llm.invoke(prompt)
    answer = answer.strip()

    if "NEEDS_SPECIALIST" in answer.upper():
        return AIResponse(should_ai_answer=False)

    return AIResponse(should_ai_answer=True, answer=answer)

@app.get("/health")
def health():
    return {"status": "ok"}