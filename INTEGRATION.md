# Sahtech AI Service — Integration Guide

The AI service is a standalone FastAPI app with two endpoints. All apps
(Flutter, React) reach it **through the NestJS backend** — NestJS forwards the
user's `Authorization` header and proxies the request. The AI service never
talks to phones or browsers directly.

    Flutter / React  ->  NestJS  ->  AI service (this repo)  ->  Ollama (LLM)

## Running the service

    docker start ollama
    docker exec -it ollama ollama pull gemma2:2b
    docker exec -it ollama ollama pull qwen2.5:1.5b
    cp .env.example .env        # set JWT_SECRET to MATCH the NestJS secret
    docker compose run --rm ai-service python ingest.py   # build knowledge base
    docker compose up -d
    curl http://localhost:8001/health

Service listens on port 8001.

## Authentication

Every request to /ask and /analyze requires:  Authorization: Bearer <JWT>

- Same token NestJS already issues. NestJS must FORWARD the Authorization header.
- AI decodes it with the shared JWT_SECRET, reads the `role` claim
  (patient | specialist | admin; defaults to patient if missing).
- Role controls the response. Apps do NOT send role separately.
- Missing/invalid token -> 401.

## Endpoint 1 — POST /ask  (chat)

Request:
    { "message": "j'ai mal à l'épaule depuis 3 jours", "language": "fr" }
- message (required): question in Arabic / French / English.
- language (optional): hint only; service auto-detects and that wins.

Response:
    {
      "answer": "...",
      "is_medical": true,
      "detected_language": "fr",
      "is_sensitive": false
    }
- answer: reply. PATIENT -> ends with a specialist recommendation.
          SPECIALIST -> technical tone, no recommendation.
- is_medical: false -> answer is a polite refusal (non-medical question).
- detected_language: ar / fr / en. Use for text direction (RTL for ar).
- is_sensitive: true -> emergency red flag; answer starts with a ⚠️ banner.

Latency ~10-20s on CPU. Show a loading state.

## Endpoint 2 — POST /analyze  (movement report)

Takes the NUMERIC results of the on-device shoulder movement analysis and
returns a written clinical interpretation. The AI only interprets — it never
computes or changes the numbers.

Request:
    {
      "patient_name": "Jean-Dominique Morel",
      "body_part": "épaule",
      "session_date": "2026-05-22",
      "affected_side": "left",
      "metrics": {
        "flexion":          { "left":142,"right":168,"delta_left":12,"delta_right":2,"unit":"°","reference":180 },
        "extension":        { "left":38, "right":55, "delta_left":5, "delta_right":1,"unit":"°","reference":60  },
        "abduction":        { "left":110,"right":165,"delta_left":8, "delta_right":0,"unit":"°","reference":180 },
        "adduction":        { "left":28, "right":40, "delta_left":3, "delta_right":0,"unit":"°","reference":45  },
        "rotation_externe": { "left":45, "right":80, "delta_left":-2,"delta_right":1,"unit":"°","reference":90  },
        "rotation_interne": { "left":55, "right":70, "delta_left":4, "delta_right":0,"unit":"°","reference":70  }
      },
      "movement_quality": {
        "controle_moteur": "EXCELLENT",
        "compensations": "MODÉRÉ",
        "dyskinesie_scapulaire": "SUIVI"
      },
      "symmetry_score": 89,
      "recovery_score": 76,
      "mobility_pct": 82,
      "pain_pct": 14,
      "pain_scale_eva_change": -2,
      "language": "fr"
    }

- affected_side: "left" or "right" (the injured shoulder).
- metrics: 6 movements (flexion, extension, abduction, adduction,
  rotation_externe, rotation_interne). Each has left, right, per-side deltas,
  unit, and a healthy reference value.
- movement_quality: category strings (EXCELLENT / BON / MODÉRÉ / SUIVI / FAIBLE).
- language: ar / fr / en for the generated report.

Response:
    {
      "summary": "...",
      "detailed_interpretation": "...",
      "correlation": "...",
      "recommendation": "...",          // PATIENT ONLY; null for specialists
      "metrics_passthrough": { ... },   // all input numbers echoed back, unchanged
      "detected_language": "fr",
      "role": "patient"
    }

UI usage:
- Numbers (angles, scores, badges) -> render from metrics_passthrough (measured, never AI-generated).
- Prose (summary, detailed_interpretation, correlation) -> AI-written text blocks.
- recommendation -> render the box only when non-null.

The React component MovementReport.jsx (web repo) renders this exact shape.
NestJS should SAVE each report (e.g. a MovementReport table) so specialists can
view it later on the web app.

## Roles — the rule that matters

Same request, response differs by the JWT role:

                       PATIENT                              SPECIALIST
  /ask answer          full + specialist recommendation     technical, no recommendation
  /analyze recommendation   filled                          null

NestJS just forwards the real user token; the AI reads the role.

## Quick test (no NestJS needed)

    TOKEN=$(python3 -c "import jwt; print(jwt.encode({'role':'patient','sub':'t'}, 'testing-secret-not-for-production', algorithm='HS256'))")
    curl -X POST http://localhost:8001/ask \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"message":"what is diabetes"}'

See tests/ for ready scripts (test.sh, test_analyze.sh) and a browser console (test.html).
Interactive API docs: http://localhost:8001/docs
