import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

# Init OpenAI and embedder
client = OpenAI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents and their embeddings
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Fit NearestNeighbors index
index = NearestNeighbors(n_neighbors=3, metric="euclidean")
index.fit(embeddings)

# Retrieve relevant documents
def retrieve(query, k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.kneighbors(query_emb, n_neighbors=k)
    return [documents[i] for i in indices[0]]

# Generate response
def generate_answer(query):
    relevant_chunks = retrieve(query)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using the context below.
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("[INFO] 🔎 RAG Q&A Chatbot Ready!")
    while True:
        user_q = input("\nAsk your question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        answer = generate_answer(user_q)
        print("\n[Answer]:", answer)
