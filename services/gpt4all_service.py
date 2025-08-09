from gpt4all import GPT4All

# Load local model (download once and reuse)
llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")

def ask_local_llm(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = llm.generate(prompt, max_tokens=500)
    return output.strip()
