from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load env (works locally with .env; on Render, values come from Dashboard)
load_dotenv()

# --- Load PDF ---
BASE_DIR = os.path.dirname(__file__)
pdf_path = os.path.join(BASE_DIR, "general_handbook_serving_in_the_church_of_jesus_christ_of_latter_day_saints.pdf")
pages = PyPDFLoader(pdf_path).load()

# --- Split ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(pages)

# --- Embeddings (keep consistent with your prebuilt faiss_index) ---
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# --- Vector store ---
index_dir = os.path.join(BASE_DIR, "faiss_index")
if os.path.exists(index_dir):
    vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print("Loaded existing FAISS index.")
else:
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(index_dir)
    print(f"Built and saved FAISS index with {len(chunks)} chunks.")

# --- LLM ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

prompt_template = """
You are a careful and respectful assistant helping interpret official LDS Church policy materials.
Use only the information in the provided context to answer the question.
If the answer is not clearly supported by the document, reply: "I don't know."

### Guidelines
- Base all answers strictly on the context. Do not speculate, infer doctrine, or give personal interpretation.
- When quoting or paraphrasing, maintain the respectful and neutral tone of official Church communications.
- If multiple sources or policies differ, summarize each faithfully without assuming which is "correct."
- Avoid doctrinal commentary, personal opinion, or speculation about Church leaders’ intent.
- When unsure, encourage the user to consult the official Handbook or local priesthood leadership for clarification.

### Output
Provide a concise, respectful answer in 1–4 sentences.
If appropriate, include a short bullet list summarizing key points or policy steps.

### Context
{context}

### Question
{question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

crc = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    res = crc.invoke({"question": question})
    answer = (res.get("answer") or res.get("result") or "").strip()

    sources = []
    for d in (res.get("source_documents") or [])[:5]:
        m = getattr(d, "metadata", {}) or {}
        sources.append({
            "source": m.get("source") or m.get("file_path") or m.get("url") or "Unknown",
            "page_label": str(m.get("page_label", m.get("page", "N/A"))),
            "snippet": (getattr(d, "page_content", "") or "")[:600],
        })
    return jsonify({"answer": answer, "sources": sources})

# Locally:
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
