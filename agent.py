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
# ==========================================
# 3. FASTAPI APPLICATION SETUP
# ==========================================
app = FastAPI(title="HR Assistant RAG Chatbot")

# Simple HTML Chat Interface
CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR Assistant - Training Academy</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 h-screen flex flex-col justify-between">
    <!-- Header -->
    <header class="bg-indigo-600 text-white p-4 shadow-md flex justify-between items-center">
        <h1 class="text-lg font-semibold">HR Assistant RAG Chatbot</h1>
        <span class="text-xs bg-indigo-500 px-2 py-1 rounded-full">Online</span>
    </header>

    <!-- Chat Box -->
    <main id="chat-container" class="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl w-full mx-auto">
        <div class="flex items-start space-x-2">
            <div class="bg-white border border-slate-200 text-slate-800 p-3 rounded-lg shadow-sm max-w-lg">
                Hello! I am your HR Assistant. Ask me anything about our training academy policies.
            </div>
        </div>
    </main>

    <!-- Input Form -->
    <footer class="bg-white border-t border-slate-200 p-4 shadow-lg">
        <form id="chat-form" class="max-w-3xl mx-auto flex gap-2">
            <input 
                id="message-input" 
                type="text" 
                placeholder="Type your HR policy question here..." 
                required
                class="flex-1 border border-slate-300 rounded-lg px-4 py-2 focus:outline-none focus:border-indigo-500"
            >
            <button 
                type="submit" 
                class="bg-indigo-600 text-white px-5 py-2 rounded-lg font-medium hover:bg-indigo-700 transition"
            >
                Send
            </button>
        </form>
    </footer>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const chatForm = document.getElementById('chat-form');
        const messageInput = document.getElementById('message-input');
        const sessionId = "web-user-" + Math.random().toString(36.substring(2, 9));

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = messageInput.value.trim();
            if (!text) return;

            // Append User Message
            appendMessage(text, 'user');
            messageInput.value = '';
            messageInput.disabled = true;

            // Show typing indicator / loading state
            const loadingId = appendMessage('Thinking...', 'assistant', true);

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, message: text })
                });
                const data = await res.json();
                
                // Remove loading and append actual response
                document.getElementById(loadingId).remove();
                if (res.ok) {
                    appendMessage(data.response, 'assistant');
                } else {
                    appendMessage(data.detail || 'Something went wrong.', 'assistant');
                }
            } catch (err) {
                document.getElementById(loadingId).remove();
                appendMessage('Network error. Please try again.', 'assistant');
            } finally {
                messageInput.disabled = false;
                messageInput.focus();
            }
        });

        function appendMessage(text, sender, isLoading = false) {
            const id = 'msg-' + Math.random().toString(36).substring(2, 9);
            const isUser = sender === 'user';
            const wrapper = document.createElement('div');
            wrapper.id = id;
            wrapper.className = `flex items-start space-x-2 ${isUser ? 'justify-end' : ''}`;
            
            wrapper.innerHTML = `
                <div class="${isUser ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-800'} p-3 rounded-lg shadow-sm max-w-lg ${isLoading ? 'italic text-slate-400' : ''}">
                    ${escapeHtml(text)}
                </div>
            `;
            chatContainer.appendChild(wrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return id;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_chat_ui():
    return CHAT_UI_HTML
