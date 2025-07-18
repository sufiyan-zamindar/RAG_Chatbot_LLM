import sys
print("Using Python:", sys.executable)

from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pdf_loader import process_pdfs

chunks_with_meta = process_pdfs()
documents = [chunk for chunk, _ in chunks_with_meta]
metadata = [meta for _, meta in chunks_with_meta]

if not documents:
    raise ValueError("🚫 No documents found. Put PDFs in /docs!")

print(f"[INFO] Loaded {len(documents)} text chunks")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=True
)

dimension = embeddings.shape[1] if len(embeddings.shape) == 2 else embeddings[0].shape[0]
print(f"[INFO] Embedding dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"[INFO] FAISS index created with {index.ntotal} vectors")

faiss.write_index(index, "docs.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(documents, f)
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("[INFO] Embeddings + index + metadata saved successfully!")
