import re
from nlu.contracts import PreprocessResult

def preprocess(text: str, session_id: str = "default-session") -> PreprocessResult:
    raw_text = text
    # Normalize text: lowercase and strip extra whitespace
    cleaned = text.lower().strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return PreprocessResult(
        raw=raw_text,
        clean_text=cleaned,
        language="en",
        session_id=session_id
    )