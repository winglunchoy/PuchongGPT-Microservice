import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.together import TogetherLLM
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine, CondenseQuestionChatEngine # Or SimpleChatEngine

# Load environment variables from .env file (for local development)
load_dotenv()

# Llama Cloud Specifics
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_PROJECT_NAME = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")
LLAMA_CLOUD_ORGANIZATION_ID = os.getenv("LLAMA_CLOUD_ORGANIZATION_ID")
LLAMA_CLOUD_INDEX_NAME = os.getenv("LLAMA_CLOUD_INDEX_NAME", "Smart Puchong")

# API Keys for your external models
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Specifics
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-2-27b-it")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- LlamaIndex Global Objects ---
# These will be initialized once during lifespan
global_llama_cloud_index = None 
global_llama_cloud_query_engine = None
global_llama_cloud_retriever = None # Add this to store the retriever

class QueryRequest(BaseModel):
    query: str
    session_id: str # ession ID to differentiate conversations

class SourceNode(BaseModel):
    text: str
    score: float
    metadata: dict

class RagResponse(BaseModel):
    query: str
    response: str
    source_nodes: list[SourceNode]

session_chat_engines = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_llama_cloud_index
    global global_llama_cloud_query_engine
    global global_llama_cloud_retriever

    missing_vars = []
    if not LLAMA_CLOUD_API_KEY:
        missing_vars.append("LLAMA_CLOUD_API_KEY")
    if not LLAMA_CLOUD_ORGANIZATION_ID:
        missing_vars.append("LLAMA_CLOUD_ORGANIZATION_ID")
    if not TOGETHER_API_KEY:
        missing_vars.append("TOGETHER_API_KEY")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")

    if missing_vars:
        error_msg = (
            "CRITICAL ERROR: The following environment variables must be set: " +
            ", ".join(missing_vars)
        )
        print(error_msg)
        raise ValueError(error_msg)

    print("Initializing RAG service components (Llama Cloud Index, Together AI, OpenAI Embeddings)...")
    try:
        # Set the LLM to Together AI
        Settings.llm = TogetherLLM(model=LLM_MODEL, api_key=TOGETHER_API_KEY)

        # Set the Embedding Model to OpenAI text-embedding-3-small
        Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        
        # Connects to your managed index on Llama Cloud ONCE
        global_llama_cloud_index = LlamaCloudIndex(
            name=LLAMA_CLOUD_INDEX_NAME,
            project_name=LLAMA_CLOUD_PROJECT_NAME,
            organization_id=LLAMA_CLOUD_ORGANIZATION_ID,
            api_key=LLAMA_CLOUD_API_KEY,
        )

        print(f"Connected to Llama Cloud Index: {LLAMA_CLOUD_INDEX_NAME}")

        # Get the global query engine and retriever ONCE
        global_llama_cloud_query_engine = global_llama_cloud_index.as_query_engine(
            similarity_top_k=3 # Default number of documents to retrieve
        )
        global_llama_cloud_retriever = global_llama_cloud_query_engine.retriever

        print("RAG service components initialized successfully.")

    except Exception as e:
        print(f"ERROR during RAG service initialization: {e}")
        import traceback
        traceback.print_exc()
        raise

    yield # FastAPI application starts processing requests

    # --- Shutdown logic ---
    print("Shutting down RAG service components...")
    print("RAG service components shut down.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title="Smart Puchong Microservices",
    description="API for connecting Large Language Models",
    version="1.0.0",
    lifespan=lifespan
)

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root Endpoint (for health check/info) ---
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Microservice! Use /query to ask questions."}

# --- RAG Query Endpoint ---
@app.post("/query", response_model=RagResponse)
async def rag_query(request: QueryRequest):
    """
    Handles Retrieval Augmented Generation (RAG) queries with session memory.
    - Takes a user query and a session ID.
    - If a session exists, continues the conversation.
    - If new session, creates a new chat engine with memory.
    - Uses LlamaIndex (via LlamaCloudIndex) to retrieve relevant data.
    - Sends query + context + chat history to Together AI LLM for response generation.
    - Returns the LLM's answer and the source nodes.
    """
    global session_chat_engines
    global global_llama_cloud_retriever # Access the globally initialized retriever

    # Ensure core components are initialized
    if global_llama_cloud_retriever is None:
        raise HTTPException(status_code=503, detail="RAG service core components not initialized yet. Please check server logs.")

    # Get or create a chat engine for the session
    if request.session_id not in session_chat_engines:
        print(f"Creating new chat session for ID: {request.session_id}")
        try:
            # We now use the globally initialized retriever to create the chat engine
            # No need to re-initialize LlamaCloudIndex or query_engine here
            
            # Initialize memory for the chat engine
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000) # Adjust token_limit as needed

            session_chat_engines[request.session_id] = ContextChatEngine.from_defaults(
                retriever=global_llama_cloud_retriever, # Pass the globally initialized retriever
                memory=memory,
                chat_history=[], 
            )
            print(f"Initialized chat engine for session {request.session_id}")
        except Exception as e:
            print(f"Error creating chat engine for session {request.session_id}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize chat session: {e}")

    chat_engine = session_chat_engines[request.session_id]

    try:
        response = await chat_engine.achat(request.query)

        source_nodes_list = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            source_nodes_list = [
                SourceNode(
                    text=node.text,
                    score=node.score,
                    metadata=node.metadata
                )
                for node in response.source_nodes
            ]

        return RagResponse(
            query=request.query,
            response=str(response),
            source_nodes=source_nodes_list
        )
    except Exception as e:
        print(f"Error processing chat query '{request.query}' for session {request.session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during chat: {e}. Please check service logs.")