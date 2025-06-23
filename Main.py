# 📘 Mini LLM PDF QA App
import os
import re
import fitz  # PyMuPDF
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from self_model import self_mod  # Assuming self_mod is defined in a separate file
# ---------------------- Utility Functions ----------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9.?! \n]+", "", text)
    return text.lower()

def tokenize(text):
    return text.split()  # Custom simple whitespace tokenizer

def extract_certificates(text):
    match = re.search(r"(?i)(certifications?|courses|achievements)[\s\S]{0,500}", text)
    return match.group(0) if match else "No certificate section found."

def get_relevant_chunk(text, question, chunk_size=300):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    similarities = cosine_similarity(vectors[-1], vectors[:-1])
    return chunks[similarities.argmax()]

def create_vocab(tokens):
    vocab = sorted(set(tokens))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(tokens, stoi):
    return [stoi[t] for t in tokens if t in stoi]

def decode(indices, itos):
    return ' '.join([itos[i] for i in indices])

# ---------------------- Dataset Definition ----------------------
class TokenDataset(Dataset):
    def _init_(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def _len_(self):
        return len(self.tokens) - self.block_size

    def _getitem_(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size])
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1])
        return x, y

# ---------------------- Transformer Model ----------------------
class MiniTransformer(nn.Module):
    def _init_(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super()._init_()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T).unsqueeze(0).expand(B, T).to(x.device)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)

# ---------------------- Training Logic ----------------------
def train_model(model, dataloader, epochs=3, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    return losses

# ---------------------- Inference ----------------------
def generate_answer(model, stoi, itos, prompt, max_len=50):
    model.eval()
    tokens = tokenize(prompt)
    input_ids = torch.tensor([stoi.get(t, 0) for t in tokens]).unsqueeze(0).to(device)

    for _ in range(max_len):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

    return decode(input_ids[0].tolist(), itos)

# ---------------------- Streamlit UI ----------------------

# 📘 Mini LLM PDF QA App using HuggingFace Transformers
import os
import fitz  # PyMuPDF
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import re

# ---------------------- Utility Functions ----------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def preprocess_text(text):
    return re.sub(r"\s+", " ", text).strip()

def get_most_relevant_chunk(text, question, chunk_size=450):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    sims = cosine_similarity(vectors[-1], vectors[:-1])
    return chunks[sims.argmax()]

def extract_certificates(text):
    match = re.search(r"(?i)(certifications?|courses|achievements)[\s\S]{0,500}", text)
    return match.group(0).strip() if match else "No certificate section found."

import os
import fitz  # PyMuPDF
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------- PDF Reading & Text Cleaning --------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return re.sub(r'\s+', ' ', text.strip())

# -------------------- Smart Context Retrieval --------------------
def retrieve_context(text, query, max_len=450):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    chunks.append(current_chunk.strip())

    tfidf = TfidfVectorizer()
    embeddings = tfidf.fit_transform(chunks + [query])
    cosine_scores = cosine_similarity(embeddings[-1], embeddings[:-1])
    best_idx = cosine_scores.argmax()
    return chunks[best_idx]

# -------------------- Deep Q&A with Transformers --------------------
class ResumeQA:
    def _init_(self, model_name=self_mod):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer_question(self, context, question):
        tokens = self.tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model(**tokens)
        start = torch.argmax(output.start_logits)
        end = torch.argmax(output.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0][start:end])
        )
        confidence = torch.softmax(output.start_logits, dim=1)[0][start].item()
        return answer.strip(), confidence

# -------------------- Streamlit UI --------------------
st.set_page_config("Resume LLM Q&A", layout="centered")
st.title("📄 Smart Resume Q&A Assistant")
st.caption("Ask intelligent questions about any uploaded resume!")

uploaded_file = st.file_uploader("📤 Upload Resume PDF", type="pdf")
question = st.text_input("💬 Ask a question (e.g., What are the skills? Where did they intern?)")

if uploaded_file and question:
    with st.spinner("Reading PDF and analyzing..."):
        # Save temporarily and extract content
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.read())
        resume_text = extract_text("temp_resume.pdf")
        context = retrieve_context(resume_text, question)

        # Perform Q&A
        qa_model = ResumeQA()
        answer, confidence = qa_model.answer_question(context, question)

    # Show results
    st.subheader("🤖 Answer")
    st.markdown(f"*Answer:* {answer}")
    st.caption(f"📊 Confidence: {confidence:.2f}")

    # Optional highlight section
    st.subheader("📌 Matched Resume Snippet")
    st.info(context)

    st.download_button("⬇ Download Answer", answer, file_name="resume_answer.txt")