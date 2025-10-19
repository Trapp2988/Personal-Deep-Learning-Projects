#!/usr/bin/env python
# coding: utf-8

# # **LDS Church Policy Handbook RAG, LangChain and LLM**
# ## **Author:** Tanner Rapp
# ### **Policy Source:** https://www.churchofjesuschrist.org/study/manual/general-handbook?lang=eng

# ### About this Program
# ---

# ### Libraries
# ---

# In[19]:


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# In[2]:


#os.chdir(r"C:\Users\tanne\Church Policy RAG") #Only run if locally for finding api key!


# ### Load the Policy PDF
# ---

# In[3]:


BASE_DIR = os.path.dirname(__file__)  # folder where app.py lives
pdf_path = os.path.join(BASE_DIR, "general_handbook_serving_in_the_church_of_jesus_christ_of_latter_day_saints.pdf")

loader = PyPDFLoader(pdf)
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages.")


# In[4]:


# Showing Introduction from the General Handbook
print(pages[17])


# ### Split
# ---

# In[5]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]) # The splitter goes down the list in order — it first tries to split on paragraph breaks (\n\n), 
                                        # then line breaks (\n), then spaces ( ), and finally anywhere ("") if needed.

chunks = splitter.split_documents(pages)
print(len(chunks))


# ### Embed & Store
# ---

# In[6]:


# Initialize the embedding model — converts each text chunk into a numerical embedding used for semantic similarity search in RAG.
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",  # Model turns text chunks into numeric vectors so you can search for relevant content.
    encode_kwargs={"normalize_embeddings": True}
)


# In[7]:


#Save into vector database or load FAISS index ---- 
index_dir = "faiss_index"

if os.path.exists(index_dir):
    vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print("Loaded existing FAISS index.")
else:
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(index_dir)
    print(f"Built and saved FAISS index with {len(chunks)} chunks.")


# ### LLM Construction
# ---

# In[8]:


import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

# 1) Confirm where the notebook is running
print("CWD:", os.getcwd())

# 2) Point directly to your .env (adjust path if needed)
env_path = Path(r"C:\Users\tanne\Church Policy RAG\.env")
print("ENV exists:", env_path.exists(), "->", env_path)

# 3) Peek at raw bytes & text (helps catch encoding issues)
if env_path.exists():
    print("First 4 bytes:", env_path.read_bytes()[:4])
    try:
        print("Text preview:\n", env_path.read_text(encoding="utf-8"))
    except Exception as e:
        print("UTF-8 read error:", e)

# 4) Parse without setting (diagnostic)
print("dotenv_values:", dotenv_values(dotenv_path=env_path, encoding="utf-8"))

# 5) Actually load, force override, and show result
loaded = load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8")
print("load_dotenv returned:", loaded)
print("OPENAI_API_KEY ->", os.getenv("OPENAI_API_KEY"))


# In[9]:


# Accessing API Key from local storage
load_dotenv(encoding="utf-8-sig", override=True)


# In[10]:


llm = ChatOpenAI(
    model="gpt-4o-mini",     
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY") 
)


# In[11]:


# Prompt Template
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

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


# In[12]:


# Give the chatbot a memory
memory = ConversationBufferMemory(
    memory_key = "chat_history", 
    return_messages = True,
    output_key="answer" )

history = []


# In[13]:


crc = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": PROMPT}  # your LDS policy prompt
)


# ### Retrevial & Generation
# ---

# In[14]:


def prompt():
    while True:
        query = input("Ask a question about LDS Church Policy:  ").strip()

        # Exit conditions
        if query.lower() in ['exit', 'bye','goodbye']:
            print("Goodbye!")
            break

        # Run query through your retrieval chain
        response = crc.invoke({"question": query}, {"chat_history": history})

        # Print results safely
        print("\n\n---------------------------------------------------------------------------------\n\n", response['answer'])
        print("\nSource Page #: ",
            response['source_documents'][0].metadata.get('page_label', 'N/A'),
            "  ⬇️ See below."
        )
        print("\nSource: ", response['source_documents'][0].page_content)


# In[15]:


# Enter a prompt! 
#prompt()


# ### Web Interface
# ---

# In[16]:


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

    # No history: invoke the chain fresh each time
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



# In[17]:


# Run only if locally
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


# In[18]:


print("Seems good to me!")

