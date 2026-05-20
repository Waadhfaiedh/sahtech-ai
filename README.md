# Sahtech AI Service

Conversational medical assistant for the Sahtech platform — multilingual (Arabic, French, English), role-aware (patient/specialist), and grounded in a private medical knowledge base via RAG.

## Stack

- **FastAPI** — HTTP service on port 8001
- **Ollama** — local LLM runtime
  - `gemma2:2b` — answer generation (~10-15s on CPU)
  - `qwen2.5:1.5b` — medical/non-medical classifier
- **ChromaDB** — vector store for medical document embeddings
- **LangChain** — RAG orchestration
- **JWT** — auth, role extracted from `role` claim

## Setup

### 1. Start Ollama and pull models

```bash
docker run -d --name ollama -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -e OLLAMA_KEEP_ALIVE=24h \
  ollama/ollama

docker exec -it ollama ollama pull gemma2:2b
docker exec -it ollama ollama pull qwen2.5:1.5b
```

### 2. Add medical documents

Place PDF/DOCX/PPTX/ODT/TXT files into `./documents/`.

### 3. Build the vector store

```bash
docker compose run --rm ai-service python ingest.py
```

### 4. Start the service

```bash
docker compose up -d
curl http://localhost:8001/health
```

## Testing

- **Browser console:** open `test.html` directly
- **CLI:** `./test.sh patient "ما هي أعراض ارتفاع ضغط الدم"`
- **API docs:** `http://localhost:8001/docs`

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `JWT_SECRET` | `testing-secret-not-for-production` | JWT signing key (must match NestJS) |
| `OLLAMA_MODEL` | `gemma2:2b` | Answer model |
| `OLLAMA_CLASSIFIER_MODEL` | `qwen2.5:1.5b` | Classifier model |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama endpoint |

## API

`POST /ask`
```json
{ "message": "..." }
```
Headers: `Authorization: Bearer <JWT>`

Response:
```json
{
  "answer": "...",
  "is_medical": true,
  "detected_language": "fr",
  "is_sensitive": false
}
```