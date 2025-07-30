# 🤖 RAG Q&A Chatbot on Custom PDFs using FAISS and OpenAI

This project is a **Retrieval-Augmented Generation (RAG) Q&A Chatbot** that allows users to ask questions based on the content of 
**custom PDF files**. It uses **SentenceTransformers** to convert text into embeddings, **FAISS** for fast vector similarity search, 
and **OpenAI's GPT-4o model** to generate high-quality answers using the retrieved context.

---

## 🧠 How It Works

The pipeline consists of 4 main stages:

1. **PDF Loading & Chunking**  
   PDFs are loaded and split into overlapping text chunks using NLTK.

2. **Embedding & Indexing**  
   Each chunk is transformed into vector embeddings using `all-MiniLM-L6-v2` and stored in a FAISS vector index.

3. **Retrieval**  
   For any user question, the top-k most relevant chunks are retrieved from FAISS.

4. **Generation**  
   The retrieved context is passed along with the question to GPT-4o, which generates a coherent, context-aware answer.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

'''bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot


### 2. Set Up a Virtual Environment
- python -m venv env
- .\env\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add Your PDFs
- Put all your PDF files in the /docs folder.

---

## Run the Pipeline

### Step 1: Load PDFs and Generate Embeddings
- python embedder.py

### Step 2: Start the Chatbot

- python chatbot.py

### OpenAI API Key (Required)
Make sure you have an OpenAI API key. Set tihs in your environment:

- setx OPENAI_API_KEY "your_openai_key"

---

## Example Use Case

You upload 3 research papers in PDF form. You then ask:

- “What are the key findings of the second paper?”

- “Who proposed the ABC method in these documents?”

- “Explain the experimental results.”

The chatbot retrieves the best chunks and gives accurate answers using GPT-4o.

---

## Benefits

- Answer questions based on your custom PDFs

- Combine retrieval with generation for accurate results

- Fast search using FAISS

- Easy to extend or plug into a UI (e.g., Streamlit, Gradio)

---

## Contact
- Made by: Sufiyan Zamindar
- GitHub: https://www.github.com/sufiyanzamindar
- linkedin : https://www.linkedin.com/in/sufiyan012/
- Email: sufiyanzmaindar012@gmail.com

## License
This project is open-source under the MIT License.
