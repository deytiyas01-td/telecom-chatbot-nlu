from pydantic import BaseModel
from typing import Optional, Any

class PreprocessResult(BaseModel):
    raw: str
    clean_text: str
    language: Optional[str] = "en"
    session_id: Optional[str] = "default-session"

class IntentResult(BaseModel):
    name: str
    confidence: float
    top_k: list[tuple[str, float]] = []
    decision: str = "accept"

class EntityItem(BaseModel):
    type: str
    value: Any
    raw: str
    span: list[int]
    source: str = "pattern"
    confidence: float = 1.0