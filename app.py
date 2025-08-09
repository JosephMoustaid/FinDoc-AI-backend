from flask import Flask, request, jsonify
import os
from services.pdf_ingest import extract_text_from_pdf, store_pdf_embeddings, model, index
from services.qa_service import retrieve_chunks, ask_llm

app = Flask(__name__)
pdf_chunks = []

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global pdf_chunks
    file = request.files['file']
    path = os.path.join("data/uploads", file.filename)
    file.save(path)
    text = extract_text_from_pdf(path)
    pdf_chunks = store_pdf_embeddings(text)
    return jsonify({"status": "PDF processed"}), 200

@app.route('/qa', methods=['GET'])
def qa():
    query = request.args.get("q")
    retrieved = retrieve_chunks(query, pdf_chunks, index, model)
    answer = ask_llm(query, " ".join(retrieved))
    return jsonify({"answer": answer})

@app.route('/view-pdf-text', methods=['GET'])
def view_pdf_text():
    return jsonify({"chunks": pdf_chunks})
    
if __name__ == '__main__':
    app.run(debug=True)
