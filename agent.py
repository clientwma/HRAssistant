import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
HR_INDEX_NAME = os.getenv("HR_INDEX_NAME", "hr-policies")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")

# ==========================================
# 2. RAG PIPELINE INITIALIZATION
# ==========================================
llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.2,
    max_tokens=300,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(HR_INDEX_NAME)

# 2a. String Formatter function
def format_docs(docs) -> str:
    return "\n\n".join(
        f"- {d.metadata.get('text', '')}: {d.page_content}"
        for d in docs
    )

# 2b. Define retriever embeddings model and vector store
embeddings_model = OpenAIEmbeddings(
    model=OPENAI_EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings_model,
    text_key="text",
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # return top 2 most similar chunks
)

# 2c. Create PROMPT TEMPLATE
retrieval_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are HR assistant for a Training academy. "
     "Answer the user's question in strictly professional way complying with HR policies using ONLY the context provided below. "
     "If the context does not contain a relevant answer, say you don't have a "
     "matching recommendation - never cite references to the policies from other companies or trainings."
     ),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}"
     ),
])

# 2d. Tie it together using langchain
parser = StrOutputParser()
rag_chain = (
    {
        "context": retriever | format_docs,  
        "question": RunnablePassthrough()    
    }
    | retrieval_prompt_template  
    | llm                        
    | parser                     
)

# ==========================================
# 3. FASTAPI APPLICATION SETUP
# ==========================================
app = FastAPI(title="HR Assistant RAG Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float

@app.get("/health")
def health_check():
    return {"status": "ok", "index": HR_INDEX_NAME}


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Main chat endpoint. Pass the same session_id across turns
    to maintain conversation context within a session.
    """
    logger.info(
        "Incoming request",
        extra={"session_id": request.session_id, "message_length": len(request.message)}
    )
    start_time = time.time()
    try:
        response_text = rag_chain.invoke(request.message)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Request completed",
            extra={"session_id": request.session_id, "latency_ms": latency_ms}
        )
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(
            "Agent invocation failed",
            extra={"session_id": request.session_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail="HR Assistant is temporarily unavailable. Please try again in a moment."
        )