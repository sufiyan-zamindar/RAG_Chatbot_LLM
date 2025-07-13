import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load OpenAI API key from env
client = OpenAI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load index & metadata
index = faiss.read_index("docs.index")
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

def retrieve(query, k=3):
    query_emb = embedder.encode([query])[0]
    D, I = index.search(query_emb.reshape(1, -1), k)
    return [documents[i] for i in I[0]]

def generate_answer(query):
    relevant_chunks = retrieve(query)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using the following context.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo" if you have credits
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("🔎 RAG Q&A Chatbot Ready!")
    while True:
        user_q = input("\nAsk your question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        answer = generate_answer(user_q)
        print("\n📌 Answer:", answer)
