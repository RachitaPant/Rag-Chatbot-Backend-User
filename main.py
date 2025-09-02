from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from gtts import gTTS
import boto3
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableLambda

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
AUDIO_DIR = "audio"
INDEX_NAME = "admin-files"
NAMESPACE = "uploaded-files"

os.makedirs(AUDIO_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

def get_pinecone_index():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error initializing Pinecone index: {e}")
        raise



def pinecone_retriever(query: str, k: int = 3):
    """Retrieve documents from Pinecone serverless index with hosted embeddings."""
    try:
        index = get_pinecone_index()

        # Query Pinecone with raw text
        results = index.search(
            namespace=NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": k
            }
        )

        print(f"Pinecone query results: {results}")  # Debug logging

        documents = []
        # For serverless indexes with hosted embedding models,
        # data comes under "result" -> "hits"
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


# Initialize retriever, LLM, and chains
try:
    retriever = RunnableLambda(lambda x: pinecone_retriever(x["input"]))
    
    prompt_template = """You are a helpful assistant that answers questions based solely on the provided documents. 
Use only the information from the documents to generate your response. 
If the documents do not contain the information needed to answer the question, respond with: 
"I don't have the information in the provided context."

Documents:
{context}

Question: {input}

Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
except Exception as e:
    print(f"Error initializing chain: {e}")
    raise

class Query(BaseModel):
    question: str


# Initialize Polly client
polly = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")  # Default: Mumbai
)

@app.post("/ask")
async def ask(query: Query):
    try:
        print(f"Received question: {query.question}")
        result = qa_chain.invoke({"input": query.question})
        answer = result.get("answer", "I could not generate an answer.")
        print(f"Answer: {answer}")
        print(f"Sources: {[doc.metadata for doc in result.get('context', [])]}")  # Debug logging

        # Generate audio file with Indian female voice
        audio_file = os.path.join(AUDIO_DIR, "response.mp3")
        polly_response = polly.synthesize_speech(
            Text=answer,
            OutputFormat="mp3",
            VoiceId="Raveena"  # or "Aditi"
        )
        with open(audio_file, "wb") as f:
            f.write(polly_response["AudioStream"].read())

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in result.get("context", [])],
            "audio_url": f"/audio/response.mp3"
        }
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
