from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, io, uuid, json
from dotenv import load_dotenv
import boto3
import requests

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

# =====================
# Config & Setup
# =====================
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("Missing REDIS_URL")

SESSION_TTL = 600  # seconds
MAX_TURNS = 5      # Number of previous exchanges to keep in context

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_NAME = "admin-files"
NAMESPACE = "uploaded-files"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required API keys")

# =====================
# Pinecone Retriever
# =====================
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist")
    return pc.Index(INDEX_NAME)


def pinecone_retriever(query: str, k: int = 3):
    try:
        index = get_pinecone_index()
        results = index.search(
            namespace=NAMESPACE,
            query={"inputs": {"text": query}, "top_k": k}
        )
        documents = []
        hits = results.get("result", {}).get("hits", [])
        for hit in hits:
            fields = hit.get("fields", {})
            documents.append(
                Document(
                    page_content=fields.get("chunk_text", ""),
                    metadata={
                        "id": hit.get("_id", ""),
                        "category": fields.get("category", ""),
                        "score": hit.get("_score", 0),
                    }
                )
            )
        return documents
    except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        return []


# =====================
# LLM + Prompt
# =====================
prompt_template = """
You are Lexcapital's official support assistant. 
Always speak as a knowledgeable and trusted representative of Lexcapital. 

Conversation history:
{history}

User question: {input}

Use the provided documents as your source of truth. 
Do not mention or reference the documents. 
If the documents do not contain the answer, politely respond with:
"I'm sorry, I don't have that information right now. Would you like me to connect you with our team?"

Documents:
{context}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["history", "input", "context"]
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

retriever = RunnableLambda(lambda x: pinecone_retriever(x["input"]))
combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# =====================
# Models
# =====================
class Query(BaseModel):
    session_id: str
    question: str

# =====================
# AWS Polly (TTS)
# =====================
polly = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

# In-memory audio cache
audio_cache = {}

# =====================
# Helper: Redis Memory
# =====================
def get_memory(session_id: str):
    history = RedisChatMessageHistory(session_id=f"chat:{session_id}", url=REDIS_URL)
    memory = ConversationBufferMemory(
        memory_key="history",
        chat_memory=history,
        return_messages=True,
        k=MAX_TURNS
    )
    return memory

# =====================
# Endpoints
# =====================
@app.post("/ask")
async def ask(query: Query):
    try:
        # Load session memory
        memory = get_memory(query.session_id)

        # Invoke QA chain with memory
        result = qa_chain.invoke({
            "input": query.question,
            "history": memory.load_memory_variables({}).get("history", [])
        })

        if isinstance(result, dict):
            answer = result.get("answer", "I could not generate an answer.")
            sources = [doc.metadata for doc in result.get("context", [])]
        else:
            answer = str(result)
            sources = []

        # Save current turn automatically in Redis
        memory.save_context({"input": query.question}, {"output": answer})

        # Generate TTS
        polly_response = polly.synthesize_speech(
            Text=answer,
            OutputFormat="mp3",
            VoiceId="Raveena"
        )
        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = polly_response["AudioStream"].read()

        return {
            "answer": answer,
            "sources": sources,
            "audio_url": f"/stream/{audio_id}"
        }

    except Exception as e:
        print(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.get("/stream/{audio_id}")
async def stream_audio(audio_id: str):
    if audio_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio not found")
    return StreamingResponse(io.BytesIO(audio_cache[audio_id]), media_type="audio/mpeg")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    # Use LangChain's RedisChatMessageHistory to read
    history_obj = RedisChatMessageHistory(session_id=f"chat:{session_id}", url=REDIS_URL)
    messages = history_obj.messages
    formatted_history = [{"question": msg.content if msg.type=="human" else None,
                          "answer": msg.content if msg.type=="ai" else None} 
                         for msg in messages]
    return {"session_id": session_id, "history": formatted_history}


@app.get("/start-session")
async def start_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/api/groq/stt")
async def stt(file: UploadFile = File(...)):
    content = await file.read()
    groq_api_url = "https://api/groq.com/openai/v1/audio/translations"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = requests.post(groq_api_url, headers=headers, files={"file": ("audio.wav", content, "audio/wav")})
    return response.json()
