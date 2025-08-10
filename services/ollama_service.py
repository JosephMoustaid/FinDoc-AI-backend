# services/ollama_service.py
import requests
import json

def ask_ollama(prompt, model="deepseek-r1:1.5b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    response.raise_for_status()
    # Get the raw text response
    raw = response.text
    # Find the first JSON object in the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != -1:
        try:
            data = json.loads(raw[start:end])
            return data.get("response", raw)
        except Exception:
            return raw
    return raw