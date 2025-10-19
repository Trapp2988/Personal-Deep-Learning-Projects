from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
index_dir = os.path.join(BASE_DIR, "faiss_index")

# --- Load same embedding model used to build index ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # or whatever you originally used
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
print("✅ Loaded existing FAISS index.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Prompt template and chain
prompt_template = """Your policy QA prompt here..."""
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
            "source": m.get("source") or "Unknown",
            "page_label": str(m.get("page_label", "N/A")),
            "snippet": (getattr(d, "page_content", "") or "")[:600],
        })
    return jsonify({"answer": answer, "sources": sources})
