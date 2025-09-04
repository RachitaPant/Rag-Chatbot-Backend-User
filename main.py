from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, io, uuid
from dotenv import load_dotenv
import boto3
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableLambda
import redis
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import requests 

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

# Config
INDEX_NAME = "admin-files"
NAMESPACE = "uploaded-files"

# Load env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required API keys")

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

# Initialize retriever + LLM
retriever = RunnableLambda(lambda x: pinecone_retriever(x["input"]))

prompt_template = """
You are Lexi Capital's official support assistant. 
Always speak as a knowledgeable and trusted representative of Lexi Capital. 

Use the provided documents as your source of truth when answering questions. 
Do not mention or reference the documents in your responses. 
Present information confidently as if you are explaining on behalf of Lexi Capital. 

If the documents do not contain the answer, politely respond with:
"I'm sorry, I don't have that information right now. Would you like me to connect you with our team?"

Keep your answers professional, concise, and helpful, while maintaining a supportive and approachable tone. 

Documents:
{context}

Question: {input}

Answer:
"""



PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

class Query(BaseModel):
    session_id: str
    question: str

polly = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

# In-memory cache for audio
audio_cache = {}

@app.post("/ask")
async def ask(query: Query):
    try:
        result = qa_chain.invoke({"input": query.question})
        answer = result.get("answer", "I could not generate an answer.")
        
        # Store chat in Redis with expiry
        session_key = f"chat:{query.session_id}"
        message = {
            "question": query.question,
            "answer": answer
        }
        redis_client.rpush(session_key, json.dumps(message))
        redis_client.expire(session_key, SESSION_TTL)

        # Generate audio (Polly stream → bytes)
        polly_response = polly.synthesize_speech(
            Text=answer,
            OutputFormat="mp3",
            VoiceId="Raveena"
        )

        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = polly_response["AudioStream"].read()

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in result.get("context", [])],
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
    raw_history = redis_client.lrange(session_key, 0, -1)
    history = [json.loads(msg) for msg in raw_history]
    return {"session_id": session_id, "history": history}

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
