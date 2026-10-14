import requests
import json

URL = "http://127.0.0.1:8000/chat"

print("--- Telecom Chatbot CLI ---")
print("Type your message below and press Enter. Type 'exit' or 'quit' to stop.\n")

session_id = "cli-user-01"

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        
        if not user_input.strip():
            continue

        payload = {
            "session_id": session_id,
            "message": user_input,
            "debug": False
        }

        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Bot Reply : {data['reply']}")
            print(f"Intent    : {data['state']['intent']} (Decision: {data['state']['decision']})")
            if data['entities']:
                print(f"Entities  : {data['entities']}")
            print("-" * 40)
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Connection error: {e}. Make sure your FastAPI server is running!")
        break