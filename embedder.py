from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pdf_loader import process_pdfs

# 1️⃣ Load text chunks
chunks_with_meta = process_pdfs()
documents = [chunk for chunk, meta in chunks_with_meta]
metadata = [meta for chunk, meta in chunks_with_meta]

# 2️⃣ Create embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, convert_to_tensor=False)

# 3️⃣ Store in FAISS
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4️⃣ Save index & metadata
faiss.write_index(index, "docs.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(documents, f)
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Embeddings created & index saved.")
