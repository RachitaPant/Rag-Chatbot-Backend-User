from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
import glob
from pypdf import PdfReader

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load / process documents
def load_docs():
    docs = []
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        reader = PdfReader(filepath)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        docs.append(Document(page_content=text,metadata={"source":filepath}))
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(),metadata={"source":filepath}))
    return docs

# Build or load vector DB
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        docs = load_docs()
        printf(f"Loaded {len(docs)} documents")
        for doc in docs:
            print(f"Document from {doc.metadata.get('source','unknown')}:{doc.page_content[:100]}...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)  # Smaller chunks for 8 GB RAM
        splits = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(splits, embeddings, persist_directory=VECTOR_DIR)
        vectordb.persist()
    return vectordb

#custom prompt template
prompt_template="""You are a helpful assitant that answers questions based solely on the provided documents. Use only the information from the documentsto generate your response, if the documents do not contain the information needed to answer the question, respond with :"I dont have the information in the provided context."Do not use any external knowledge or make assumptions

Documents:
{context}

Question:{question}

Answer:"""


PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

# Create retriever + LLM
vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k":3})
llm = Ollama(model="phi3:3.8b-mini-128k-instruct-q4_0")  # Use Phi-3
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,chain_type_kwargs={"prompt":PROMPT})

# API models
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        answer = qa.invoke({"query": query.question})
        return {"answer": answer["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")