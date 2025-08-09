import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index setup
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def store_pdf_embeddings(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = model.encode(chunks)
    index.add(np.array(vectors, dtype=np.float32))
    return chunks
