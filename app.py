from flask import Flask, request, jsonify
import os

from services.pdf_ingest import extract_text_from_pdf, store_pdf_embeddings, model, index
from services.qa_service import retrieve_chunks
from services.gpt4all_service import ask_local_llm , merge_metrics , ask_local_llm
import math
from concurrent.futures import ThreadPoolExecutor
import json
from prompts import FINANCIAL_METRICS_PROMPT

from services.groq_service import ask_groq_llm

executor = ThreadPoolExecutor(max_workers=1)


app = Flask(__name__)
pdf_chunks = []

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global pdf_chunks
    file = request.files['file']
    path = os.path.join("data/uploads", file.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file.save(path)
    text = extract_text_from_pdf(path)
    pdf_chunks = store_pdf_embeddings(text)
    return jsonify({"status": "PDF processed"}), 200

@app.route('/qa', methods=['GET'])
def qa():
    query = request.args.get("q")
    retrieved = retrieve_chunks(query, pdf_chunks, index, model)
    answer = ask_local_llm(query, " ".join(retrieved))
    return jsonify({"answer": answer})


def batch_chunks(chunks, char_limit):
  batches = []
  batch = []
  batch_len = 0
  for chunk in chunks:
    if batch_len + len(chunk) > char_limit and batch:
      batches.append(" ".join(batch))
      batch = []
      batch_len = 0
    batch.append(chunk)
    batch_len += len(chunk)
  if batch:
    batches.append(" ".join(batch))
  return batches


@app.route('/extract-metrics', methods=['GET'])
def extract_metrics():
    char_limit = 1000  # Adjusted to fit within the model's context window
    batches = batch_chunks(pdf_chunks, char_limit)
    all_metrics = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda text: ask_local_llm(FINANCIAL_METRICS_PROMPT, text),
            batches
        ))

    for res in results:
        try:
            # Attempt to load JSON directly
            data = json.loads(res)
            if isinstance(data, list):
                all_metrics.extend(data)
        except json.JSONDecodeError:
            # Try to salvage JSON from malformed text
            try:
                start = res.find("[")
                end = res.rfind("]") + 1
                if start != -1 and end != -1:
                    fixed_json = res[start:end]
                    data = json.loads(fixed_json)
                    if isinstance(data, list):
                        all_metrics.extend(data)
            except Exception:
                continue  # skip this batch if unrecoverable

    merged = merge_metrics(all_metrics)
    return jsonify({"metrics": merged})

@app.route('/view-pdf-text', methods=['GET'])
def view_pdf_text():
    return jsonify({"chunks": pdf_chunks})

if __name__ == '__main__':
    app.run(debug=True)
