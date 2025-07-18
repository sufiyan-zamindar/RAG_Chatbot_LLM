import os
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pickle
from nltk.data import find  # ✅ Make sure this line exists!

# Download required NLTK data
nltk.download('punkt', quiet=True)

def load_pdf(file_path):
    """Extract text from PDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    """Split into overlapping chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    total_words = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        total_words += len(words)
        current_chunk.extend(words)

        if total_words >= chunk_size:
            chunk = ' '.join(current_chunk)
            chunks.append(chunk)
            overlap_words = current_chunk[-overlap:]
            current_chunk = overlap_words.copy()
            total_words = len(overlap_words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_pdfs(folder="docs"):
    """Load & chunk all PDFs in folder."""
    all_chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            text = load_pdf(os.path.join(folder, filename))
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append((chunk, {"source": filename, "chunk_id": i}))
    return all_chunks

if __name__ == "__main__":
    chunks_with_meta = process_pdfs()
    print(f"Total chunks: {len(chunks_with_meta)}")
    print(chunks_with_meta[0])