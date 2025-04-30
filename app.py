
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load Excel file
df = pd.read_excel("Startup_Finance_Sample.xlsx")
chunks = df["Combined Text"].dropna().tolist()

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)
embedding_dim = embeddings.shape[1]

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Define search function
def get_best_answer(query, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Streamlit UI
st.set_page_config(page_title="Startup Finance Chatbot", page_icon="💬")
st.title("💬 Startup Finance AI Doubt Solver")

user_input = st.text_input("Ask a question about startup finance:")

if user_input:
    results = get_best_answer(user_input)
    st.markdown("### 📘 Answer from your notes:")
    st.write(results[0])
