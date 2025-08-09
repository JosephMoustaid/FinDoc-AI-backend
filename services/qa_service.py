import numpy as np

def retrieve_chunks(query, chunks, index, model):
    q_vec = model.encode([query])
    distances, ids = index.search(np.array(q_vec, dtype=np.float32), k=3)
    return [chunks[i] for i in ids[0]]
