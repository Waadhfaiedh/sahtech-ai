import os
import re
import jwt
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---------------- Setup ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANSWER_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CLASSIFIER_MODEL = os.getenv("OLLAMA_CLASSIFIER_MODEL", "qwen2.5:1.5b")

classifier_llm = Ollama(
    model=CLASSIFIER_MODEL,
    base_url=OLLAMA_HOST,
    temperature=0.0,
    num_predict=5,
    keep_alive="24h",
)

answer_llm = Ollama(
    model=ANSWER_MODEL,
    base_url=OLLAMA_HOST,
    temperature=0.3,
    num_predict=768,
    keep_alive="24h",
)

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

app = FastAPI(title="Sahtech AI Service")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def warmup():
    try:
        classifier_llm.invoke("YES")
        answer_llm.invoke("Hi")
        print(f"Models pre-warmed: {ANSWER_MODEL} + {CLASSIFIER_MODEL}")
    except Exception as e:
        print(f"Warmup failed (will retry on first request): {e}")

# ---------------- Schemas ----------------
class MessageRequest(BaseModel):
    message: str
    language: str | None = None

class AIResponse(BaseModel):
    answer: str
    is_medical: bool
    detected_language: str
    is_sensitive: bool = False

# ---------------- Language detection ----------------
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
FRENCH_HINTS = re.compile(
    r"\b(le|la|les|un|une|des|du|de|est|sont|j'?ai|mon|ma|mes|mal|douleur|tête|santé|"
    r"médecin|épaule|ventre|gorge|fièvre|fatigue|bonjour|merci|pourquoi|comment|quand|où)\b",
    re.IGNORECASE,
)
ACCENTS_RE = re.compile(r"[àâäéèêëîïôöùûüÿç]", re.IGNORECASE)

def detect_language(text: str) -> str:
    if len(ARABIC_RE.findall(text)) >= 2:
        return "ar"
    if ACCENTS_RE.search(text):
        return "fr"
    hits = len(FRENCH_HINTS.findall(text))
    if hits >= 2 or (hits == 1 and len(text.split()) <= 6):
        return "fr"
    return "en"

# ---------------- Prompts ----------------
LANGUAGE_NAMES = {
    "ar": "Arabic (Modern Standard Arabic / الفصحى)",
    "fr": "French",
    "en": "English",
}

LANGUAGE_STYLE = {
    "ar": (
        "اكتب الإجابة باللغة العربية الفصحى فقط، وليس بالعامية. "
        "نظّم الإجابة بفقرات قصيرة وواضحة، واستخدم نقاطًا (•) عند ذكر قائمة. "
        "تجنّب الكلمات الأجنبية إلا إذا كانت مصطلحات طبية لا بديل لها. "
        "اكتب بأسلوب طبي مهني ومفهوم."
    ),
    "fr": "Écris la réponse en français clair et professionnel. Structure-la avec des paragraphes courts et utilise des puces (•) pour les listes.",
    "en": "Write the answer in clear professional English. Structure it with short paragraphs and use bullets (•) for lists.",
}

CLASSIFIER_PROMPT = """Is the following question medical or health-related?
Medical = symptoms, diseases, treatments, medications, anatomy, nutrition, mental health, pregnancy, first aid, physiotherapy.
Not medical = greetings, programming, sports, weather, math, chit-chat.

Reply with EXACTLY one word: YES or NO.

Question: {question}
Answer:"""

ANSWER_PROMPT = """You are a medical assistant for Sahtech, a healthcare platform in Tunisia.

The user wrote in {language_name}. You MUST write your ENTIRE answer in {language_name}. Do not mix languages.

{style_instruction}

Context from medical documents:
{context}

User question: {question}

Instructions:
- Give a thorough, well-structured answer covering causes, symptoms, and general guidance.
- Use the context above when relevant; otherwise rely on accurate general medical knowledge.
- Do NOT invent specific dosages, brand names, or statistics not in the context.
{role_instructions}

Answer (in {language_name}):"""

PATIENT_TAIL_AR = """- في نهاية الإجابة، وعلى سطر جديد، أضف توصية للمريض بمراجعة الطبيب المختص المناسب. استنتج التخصص من السؤال (أمثلة: طبيب القلب لآلام الصدر، طبيب الجلدية للبشرة، طبيب العظام أو أخصائي العلاج الطبيعي للمفاصل والعضلات، طبيب الأعصاب للصداع، طبيب الجهاز الهضمي للبطن، الطبيب النفسي للصحة النفسية، طبيب النساء، طبيب الأطفال، طبيب الأنف والأذن والحنجرة، طبيب العيون، طبيب المسالك البولية، طبيب الغدد الصماء، طبيب الروماتيزم).

- التحذير الطارئ مخصص فقط للحالات الخطيرة الواضحة. أضف التحذير **فقط** إذا ذكر المريض صراحةً واحدة من هذه العلامات: ألم شديد ومفاجئ في الصدر مع ضيق تنفس، علامات سكتة دماغية، نزيف حاد لا يتوقف، فقدان الوعي، أفكار انتحارية واضحة، صعوبة شديدة في التنفس، تشنجات، حمى أعلى من 39° عند رضيع أقل من 3 أشهر.

- **لا تضف التحذير** للآلام المعتادة (آلام الظهر، آلام المفاصل المستمرة منذ أيام، صداع عادي، ألم بطن خفيف، التهاب الحلق، الزكام). في حالة الشك، لا تضف التحذير.

- إذا كان التحذير مطلوبًا، أضفه في أعلى الإجابة بالضبط هكذا: **⚠️ قد تكون هذه حالة طبية طارئة. يرجى طلب الرعاية الطبية الفورية أو التوجه إلى أقرب قسم طوارئ.**"""

PATIENT_TAIL_FR = """- À la fin de ta réponse, sur une nouvelle ligne, ajoute une recommandation de consulter un spécialiste. Déduis la spécialité de la question (cardiologue, dermatologue, orthopédiste, kinésithérapeute, neurologue, gastro-entérologue, pneumologue, psychiatre, gynécologue, pédiatre, ORL, ophtalmologue, urologue, endocrinologue, rhumatologue).

- Le warning d'urgence est réservé UNIQUEMENT aux vraies urgences vitales. Ajoute le warning **SEULEMENT** si le patient mentionne explicitement un de ces signaux : douleur thoracique intense et soudaine avec essoufflement, signes d'AVC (visage qui s'affaisse, élocution troublée, faiblesse soudaine d'un côté), saignement abondant qui ne s'arrête pas, perte de conscience, pensées suicidaires explicites, détresse respiratoire sévère, convulsions, fièvre > 39°C chez un nourrisson de moins de 3 mois.

- **N'ajoute PAS le warning** pour les douleurs courantes (mal de dos, douleur articulaire de plusieurs jours, mal de tête ordinaire, légère douleur abdominale, mal de gorge, rhume). Dans le doute, n'ajoute pas le warning.

- Si le warning est nécessaire, ajoute-le tout en haut exactement ainsi : **⚠️ Ceci peut être une urgence médicale. Consultez immédiatement un médecin ou rendez-vous aux urgences les plus proches.**"""

PATIENT_TAIL_EN = """- At the end of your answer, on a new line, add a recommendation to consult a specialist. Infer the specialty from the question (cardiologist, dermatologist, orthopedist, physiotherapist, neurologist, gastroenterologist, pulmonologist, psychiatrist, gynecologist, pediatrician, ENT, ophthalmologist, urologist, endocrinologist, rheumatologist).

- The emergency warning is RESERVED ONLY for clear life-threatening emergencies. Add the warning **ONLY** if the patient explicitly mentions one of these signs: sudden severe chest pain with shortness of breath, stroke signs, heavy bleeding that won't stop, loss of consciousness, explicit suicidal thoughts, severe respiratory distress, seizures, fever > 39°C in an infant under 3 months.

- **DO NOT add the warning** for common pains (back pain, chronic joint pain lasting days, ordinary headache, mild abdominal pain, sore throat, cold). When in doubt, do not add the warning.

- If the warning is needed, add it at the very top exactly like this: **⚠️ This may be a medical emergency. Please seek immediate medical attention or go to the nearest emergency room.**"""

PATIENT_TAILS = {"ar": PATIENT_TAIL_AR, "fr": PATIENT_TAIL_FR, "en": PATIENT_TAIL_EN}

SPECIALIST_TAILS = {
    "ar": "- اكتب الإجابة بأسلوب مهني تقني مناسب لزميل طبيب مختص. لا تضف أي توصية بمراجعة طبيب.",
    "fr": "- Rédige la réponse dans un style professionnel et technique adapté à un confrère spécialiste. N'ajoute aucune recommandation de consulter un spécialiste.",
    "en": "- Write the answer in a professional, technical tone for a fellow medical specialist. Do not add any 'consult a specialist' recommendation.",
}

REFUSAL_MESSAGES = {
    "fr": "Je suis un assistant médical et je ne peux répondre qu'à des questions liées à la santé. N'hésitez pas à me poser une question médicale.",
    "en": "I am a medical assistant and can only answer health-related questions. Please feel free to ask me a medical question.",
    "ar": "أنا مساعد طبي ولا يمكنني الإجابة إلا على الأسئلة المتعلقة بالصحة. لا تتردد في طرح سؤال طبي.",
}

# ---------------- Helpers ----------------
def get_user_role(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    role = payload.get("role", "patient").lower()
    if role not in {"patient", "specialist", "admin"}:
        role = "patient"
    return role

def is_medical_question(question: str) -> bool:
    verdict = classifier_llm.invoke(CLASSIFIER_PROMPT.format(question=question)).strip().upper()
    # Gemma sometimes wraps the answer in extra text, so check for "YES" anywhere in first 20 chars
    return "YES" in verdict[:20] and "NO" not in verdict[:5]

def detect_sensitivity(answer_text: str) -> bool:
    upper = answer_text.upper()
    return "⚠️" in answer_text or "EMERGENCY" in upper or "URGENCE" in upper or "طارئة" in answer_text

# ---------------- Endpoints ----------------
@app.post("/ask", response_model=AIResponse)
async def ask(req: MessageRequest, authorization: str | None = Header(default=None)):
    role = get_user_role(authorization)
    lang = detect_language(req.message)

    if not is_medical_question(req.message):
        return AIResponse(
            answer=REFUSAL_MESSAGES[lang],
            is_medical=False,
            detected_language=lang,
        )

    results = db.similarity_search_with_score(req.message, k=3)
    context = "\n\n".join([doc.page_content for doc, _ in results]) if results else "(no specific context retrieved)"

    role_instructions = PATIENT_TAILS[lang] if role == "patient" else SPECIALIST_TAILS[lang]

    prompt = ANSWER_PROMPT.format(
        language_name=LANGUAGE_NAMES[lang],
        style_instruction=LANGUAGE_STYLE[lang],
        context=context,
        question=req.message,
        role_instructions=role_instructions,
    )

    answer = answer_llm.invoke(prompt).strip()
    sensitive = role == "patient" and detect_sensitivity(answer)

    return AIResponse(
        answer=answer,
        is_medical=True,
        detected_language=lang,
        is_sensitive=sensitive,
    )

# ==================== MOVEMENT ANALYSIS REPORT ====================

class AnalysisRequest(BaseModel):
    patient_name: str | None = None
    body_part: str = "épaule"
    session_date: str | None = None
    affected_side: str = "left"        # "left" or "right"
    metrics: dict = {}                 # { flexion: {left,right,delta_left,delta_right,unit,reference}, ... }
    movement_quality: dict = {}
    symmetry_score: float | None = None
    recovery_score: float | None = None
    mobility_pct: float | None = None
    pain_pct: float | None = None
    pain_scale_eva_change: float | None = None
    language: str | None = None

class AnalysisResponse(BaseModel):
    summary: str
    detailed_interpretation: str
    correlation: str
    recommendation: str | None = None
    metrics_passthrough: dict
    detected_language: str
    role: str

ANALYSIS_PROMPT = """You are a medical assistant for Sahtech generating a biomechanical movement analysis report for a shoulder rehabilitation session.

Write your ENTIRE response in {language_name}. {style_instruction}

The patient's AFFECTED shoulder is the {affected_side} side. Below are the MEASURED range-of-motion values for BOTH shoulders (do NOT change these numbers, only interpret them). The healthy side is the patient's own benchmark.

{metrics_block}

Other indicators:
{indicators_block}

Write a clinical interpretation in exactly THREE sections using these exact markers:

[SUMMARY]
One or two sentences: the overall headline finding. {tone_instruction}

[INTERPRETATION]
For the affected ({affected_side}) shoulder, compare each movement to the healthy side and identify which movements show the biggest deficit (as a percentage of the healthy side). Point out the priority movement to rehabilitate. Comment on movement quality. Do NOT invent values.

[CORRELATION]
Correlate the movement findings with the recovery score and pain indicators. Explain the relationship between mobility and pain.

Output ONLY the three sections with their markers. No preamble."""

PATIENT_ANALYSIS_TONE = {
    "ar": "استخدم لغة بسيطة ومطمئنة يفهمها المريض غير المختص.",
    "fr": "Utilise un langage simple et rassurant, compréhensible par un patient non spécialiste.",
    "en": "Use simple, reassuring language understandable by a non-specialist patient.",
}
SPECIALIST_ANALYSIS_TONE = {
    "ar": "استخدم لغة طبية تقنية دقيقة مناسبة لأخصائي.",
    "fr": "Utilise un langage médical technique et précis adapté à un spécialiste.",
    "en": "Use precise, technical medical language suitable for a specialist.",
}
ANALYSIS_RECOMMENDATION = {
    "ar": "بناءً على هذا التحليل، ننصح بمراجعة أخصائي العلاج الطبيعي أو طبيب العظام لمتابعة إعادة تأهيل الكتف.",
    "fr": "Sur la base de cette analyse, nous recommandons de consulter un kinésithérapeute ou un orthopédiste pour le suivi de la rééducation de l'épaule.",
    "en": "Based on this analysis, we recommend consulting a physiotherapist or orthopedist for follow-up rehabilitation of the shoulder.",
}

def build_metrics_block(req: AnalysisRequest) -> str:
    lines = []
    for name, m in (req.metrics or {}).items():
        if not isinstance(m, dict):
            continue
        unit = m.get("unit", "")
        left = m.get("left"); right = m.get("right")
        ref = m.get("reference")
        nice = name.replace("_", " ")
        ref_txt = f", healthy reference {ref}{unit}" if ref is not None else ""
        lines.append(f"- {nice}: left {left}{unit}, right {right}{unit}{ref_txt}")
    return "\n".join(lines) if lines else "(no metrics provided)"

def build_indicators_block(req: AnalysisRequest) -> str:
    lines = []
    if req.symmetry_score is not None:
        lines.append(f"- Symmetry score: {req.symmetry_score}%")
    if req.recovery_score is not None:
        lines.append(f"- Recovery score: {req.recovery_score}/100")
    if req.mobility_pct is not None:
        lines.append(f"- Mobility: {req.mobility_pct}%")
    if req.pain_pct is not None:
        lines.append(f"- Pain level: {req.pain_pct}%")
    if req.pain_scale_eva_change is not None:
        lines.append(f"- EVA pain scale change: {req.pain_scale_eva_change}")
    if req.movement_quality:
        for k, v in req.movement_quality.items():
            lines.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(lines) if lines else "(none)"

def parse_sections(text: str) -> dict:
    sections = {"summary": "", "detailed_interpretation": "", "correlation": ""}
    mapping = {"[SUMMARY]": "summary", "[INTERPRETATION]": "detailed_interpretation", "[CORRELATION]": "correlation"}
    current = None
    for line in text.splitlines():
        if line.strip() in mapping:
            current = mapping[line.strip()]
            continue
        if current:
            sections[current] += line + "\n"
    return {k: v.strip() for k, v in sections.items()}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest, authorization: str | None = Header(default=None)):
    role = get_user_role(authorization)
    lang = req.language if req.language in ("ar", "fr", "en") else "fr"

    tone = (PATIENT_ANALYSIS_TONE if role == "patient" else SPECIALIST_ANALYSIS_TONE)[lang]
    affected_label = {"left": "left (gauche)", "right": "right (droite)"}.get(req.affected_side, req.affected_side)

    prompt = ANALYSIS_PROMPT.format(
        language_name=LANGUAGE_NAMES[lang],
        style_instruction=LANGUAGE_STYLE[lang],
        affected_side=affected_label,
        metrics_block=build_metrics_block(req),
        indicators_block=build_indicators_block(req),
        tone_instruction=tone,
    )

    raw = answer_llm.invoke(prompt).strip()
    parsed = parse_sections(raw)
    if not any(parsed.values()):
        parsed["summary"] = raw

    recommendation = ANALYSIS_RECOMMENDATION[lang] if role == "patient" else None

    return AnalysisResponse(
        summary=parsed["summary"],
        detailed_interpretation=parsed["detailed_interpretation"],
        correlation=parsed["correlation"],
        recommendation=recommendation,
        metrics_passthrough={
            "patient_name": req.patient_name,
            "session_date": req.session_date,
            "affected_side": req.affected_side,
            "metrics": req.metrics,
            "movement_quality": req.movement_quality,
            "symmetry_score": req.symmetry_score,
            "recovery_score": req.recovery_score,
            "mobility_pct": req.mobility_pct,
            "pain_pct": req.pain_pct,
            "pain_scale_eva_change": req.pain_scale_eva_change,
        },
        detected_language=lang,
        role=role,
    )

# ==================== END MOVEMENT ANALYSIS ====================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "answer_model": ANSWER_MODEL,
        "classifier_model": CLASSIFIER_MODEL,
    }