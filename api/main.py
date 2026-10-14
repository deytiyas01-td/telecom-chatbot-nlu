import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlu.preprocess import preprocess
from nlu.intent.baseline import IntentBaselineClassifier
from nlu.entities.patterns import extract_patterns

app = FastAPI(title="Telecom Chatbot NLP Pipeline", version="1.3")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

intent_classifier = IntentBaselineClassifier()
MODEL_PATH = "reports/baseline_intent.pkl"

sample_texts = [
    "what's my balance", "check my talktime", "how much balance do I have left",
    "recharge 299 on 9876543210", "do a quick recharge of 199", "top up my number with 500",
    "pay my postpaid bill", "clear my monthly bill", "pay bill for my connection",
    "how much internet data is left", "check remaining mobile data", "data balance check",
    "my internet is very slow", "network signal is dropping", "facing call drop issues",
    "activate international roaming pack", "activation of global roaming", "roaming plans for USA",
    "talk to a human agent", "connect me to customer care", "transfer call to a representative"
]

sample_labels = [
    "check_balance", "check_balance", "check_balance",
    "recharge_prepaid", "recharge_prepaid", "recharge_prepaid",
    "pay_bill", "pay_bill", "pay_bill",
    "check_data", "check_data", "check_data",
    "network_issue", "network_issue", "network_issue",
    "roaming_pack", "roaming_pack", "roaming_pack",
    "request_agent", "request_agent", "request_agent"
]

if not intent_classifier.load(MODEL_PATH):
    intent_classifier.train(sample_texts, sample_labels)
    intent_classifier.save(MODEL_PATH)

class ChatRequest(BaseModel):
    session_id: str = "default-session"
    message: str
    debug: bool = False

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "models": {
            "intent": "baseline", 
            "entities": "patterns",
            "classes": intent_classifier.classes_
        }
    }

@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    start = time.time()
    pre = preprocess(payload.message, session_id=payload.session_id)
    intent_res = intent_classifier.predict(pre.clean_text)
    entities = extract_patterns(pre.raw)
    latency = round((time.time() - start) * 1000, 2)
    
    return {
        "reply": f"Processed intent: {intent_res.name} (Confidence: {intent_res.confidence:.2f})",
        "state": {
            "intent": intent_res.name, 
            "decision": intent_res.decision,
            "top_k": intent_res.top_k
        },
        "entities": [e.dict() for e in entities],
        "latency_ms": {"total": latency}
    }