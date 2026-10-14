import re
from nlu.contracts import EntityItem

PATTERNS = {
    "phone_number": r"\b[6-9]\d{9}\b|\b\+?91[\s-]?\d{10}\b|\b\d{5}[-\s]?\d{5}\b",
    "amount": r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)|([\d,]+(?:\.\d{1,2})?)\s*(?:rupees|rs)|\b(?:rs\.?|inr|₹)\s*(\d+)\b|\b(\d+)\s*(?:rupees|rs)\b|\b(\d{2,4})\b",
    "ticket_id": r"\btkt-\d{6}\b",
    "data_quantity": r"\b\d+(?:\.\d+)?\s*(?:gb|mb|kb)/?(?:day)?\b",
}

def extract_patterns(text: str) -> list[EntityItem]:
    entities = []
    
    for match in re.finditer(PATTERNS["phone_number"], text, re.IGNORECASE):
        raw_val = match.group(0)
        entities.append(EntityItem(
            type="phone_number",
            value=raw_val.replace(" ", "").replace("-", ""),
            raw=raw_val,
            span=[match.start(), match.end()],
            source="pattern",
            confidence=1.0
        ))

    for match in re.finditer(PATTERNS["amount"], text, re.IGNORECASE):
        raw_val = match.group(0)
        # Avoid treating phone numbers or long digits as amounts
        if len(raw_val) == 10 and raw_val.isdigit() and raw_val.startswith(('6','7','8','9')):
            continue
        entities.append(EntityItem(
            type="amount",
            value=float(raw_val.replace(",", "")),
            raw=raw_val,
            span=[match.start(), match.end()],
            source="pattern",
            confidence=0.95
        ))

    for match in re.finditer(PATTERNS["ticket_id"], text, re.IGNORECASE):
        raw_val = match.group(0)
        entities.append(EntityItem(
            type="ticket_id",
            value=raw_val.upper(),
            raw=raw_val,
            span=[match.start(), match.end()],
            source="pattern",
            confidence=1.0
        ))

    for match in re.finditer(PATTERNS["data_quantity"], text, re.IGNORECASE):
        raw_val = match.group(0)
        entities.append(EntityItem(
            type="data_quantity",
            value=raw_val,
            raw=raw_val,
            span=[match.start(), match.end()],
            source="pattern",
            confidence=0.95
        ))

    return entities