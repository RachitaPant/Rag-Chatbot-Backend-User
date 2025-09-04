from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, io, uuid, json
from dotenv import load_dotenv
import boto3
import redis

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory

# =====================
# Config & Setup
# =====================
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("Missing REDIS_URL")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

SESSION_TTL = 86400
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
# Pinecone
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
        docs = []
        hits = results.get("result", {}).get("hits", [])
        for hit in hits:
            fields = hit.get("fields", {})
            docs.append({
                "page_content": fields.get("chunk_text", ""),
                "metadata": {
                    "id": hit.get("_id", ""),
                    "category": fields.get("category", ""),
                    "score": hit.get("_score", 0),
                }
            })
        return docs
    except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        return []

# =====================
# LLM + Memory
# =====================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# History-aware retriever
from langchain_core.runnables import RunnableLambda
retriever = RunnableLambda(lambda x: pinecone_retriever(x["input"]))
history_aware_retriever = create_history_aware_retriever(llm, retriever)

# Prompt with memory slot
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Lexi Capital's official support assistant. "
               "Always speak as a knowledgeable representative. "
               "Use the provided documents as your source of truth when answering questions."
               "Do not mention or reference the documents in your responses. "
               "Present information confidently as if you are explaining on behalf of Lexi Capital. "
               "If documents lack the answer, say: "
               "'I'm sorry, I don't have that information right now. "
               "Would you like me to connect you with our team?'"
               "Keep your answers professional, concise, and helpful, while maintaining a supportive and approachable tone."),
    MessagesPlaceholder("chat_history"),  
    ("user", "{input}")
])

# Document combination
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# Retrieval chain
qa_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

# Memory (Redis-backed)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# =====================
# Models
# =====================
class Query(BaseModel):
    session_id: str
    question: str

# AWS Polly
polly = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

# In-memory audio cache
audio_cache = {}

# =====================
# Endpoints
# =====================
@app.post("/ask")
async def ask(query: Query):
    try:
        session_key = f"chat:{query.session_id}"

        # Pull history from Redis
        raw_history = redis_client.get(session_key)
        history = json.loads(raw_history) if raw_history else []

        # Feed history + latest question
        result = qa_chain.invoke({
            "input": query.question,
            "chat_history": history
        })

        answer = result.get("answer", "I could not generate an answer.")

        # Update history (store both Q & A)
        history.append({"role": "user", "content": query.question})
        history.append({"role": "assistant", "content": answer})
        redis_client.set(session_key, json.dumps(history), ex=SESSION_TTL)

        # Audio (Polly)
        polly_response = polly.synthesize_speech(
            Text=answer,
            OutputFormat="mp3",
            VoiceId="Raveena"
        )

        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = polly_response["AudioStream"].read()

        return {
            "answer": answer,
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
    session_key = f"chat:{session_id}"
    raw_history = redis_client.get(session_key)
    history = json.loads(raw_history) if raw_history else []
    return {"session_id": session_id, "history": history}

@app.get("/start-session")
async def start_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}
