import openai

import numpy as np
import faiss
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def retrieve_chunks(query, chunks, index, model):
    q_vec = model.encode([query])
    distances, ids = index.search(np.array(q_vec, dtype=np.float32), k=3)
    return [chunks[i] for i in ids[0]]

def ask_llm(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return resp.choices[0].text.strip()
