import sys
import os

# Ensure the root directory is in the Python path so local modules like 'nlu' can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import time
from nlu.preprocess import preprocess
from nlu.intent.baseline import IntentBaselineClassifier
from nlu.entities.patterns import extract_patterns

# Page Configuration
st.set_page_config(
    page_title="Telecom AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize Classifier (Cached so it loads only once)
@st.cache_resource
def load_classifier():
    classifier = IntentBaselineClassifier()
    model_path = "reports/baseline_intent.pkl"
    
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
    
    if not classifier.load(model_path):
        classifier.train(sample_texts, sample_labels)
        classifier.save(model_path)
    return classifier

classifier = load_classifier()

# UI Layout
st.title("🤖 Telecom NLU Assistant")
st.markdown("Ask me about your balance, recharges, data packs, or network issues!")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your telecom assistant. How can I help you today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            meta = message["metadata"]
            st.caption(f"Intent: **{meta['intent']}** | Confidence: **{meta['confidence']*100:.0f}%**" + 
                       (f" | Entities: {', '.join([f'{e.type}: {e.value}' for e in meta['entities']])}" if meta['entities'] else ""))

# Handle User Input
if prompt := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via NLU pipeline
    start = time.time()
    pre = preprocess(prompt, session_id="streamlit-user")
    intent_res = classifier.predict(pre.clean_text)
    entities = extract_patterns(pre.raw)
    latency = round((time.time() - start) * 1000, 2)

    reply_text = f"Processed intent: {intent_res.name} (Confidence: {intent_res.confidence:.2f})"
    
    metadata = {
        "intent": intent_res.name,
        "confidence": intent_res.confidence,
        "entities": entities,
        "latency": latency
    }

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply_text, "metadata": metadata})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
        st.caption(f"Intent: **{intent_res.name}** | Confidence: **{intent_res.confidence*100:.0f}%**" + 
                   (f" | Entities: {', '.join([f'{e.type}: {e.value}' for e in entities])}" if entities else ""))