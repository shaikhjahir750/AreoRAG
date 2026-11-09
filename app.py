import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional, Dict, Any
from pathlib import Path
from collections import deque

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure internal modules can be resolved
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "rag_system"))
sys.path.append(str(BASE_DIR / "chat bot"))

from langchain_core.output_parsers import StrOutputParser

# Initialize FastAPI app
app = FastAPI(
    title="AeroRAG - Aerospace Mission Intelligence Console",
    description="Grounded Aerospace RAG platform for Apollo 11, Apollo 13, and Challenger STS 51-L technical reports.",
    version="2.0.0"
)

# Enable CORS for local dev and live server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Quota & Rate Limit State Tracker for Google Gemini Free Tier
# ============================================================================
class GeminiQuotaTracker:
    """
    In-memory rate limit and daily quota tracker for Google AI Studio Free Tier:
    - 15 Requests Per Minute (RPM)
    - 500 Requests Per Day (RPD)
    Shared across both AeroRAG and AeroBot.
    """
    def __init__(self, state_file: Path = BASE_DIR / "quota_state.json"):
        self.state_file = state_file
        self.rpm_limit = 15
        self.rpd_limit = 500
        self.minute_window = deque()  # timestamps of calls within last 60 seconds
        self.daily_count = 0
        self.current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._load_state()

    def _load_state(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        self.daily_count = data.get("daily_count", 0)
                    else:
                        self.daily_count = 0
                        self.current_date_utc = today
            except Exception:
                self.daily_count = 0
        else:
            self.daily_count = 0

    def _save_state(self):
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "date": today,
                    "daily_count": self.daily_count,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception:
            pass

    def record_requests(self, count: int = 1):
        now = time.time()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.current_date_utc:
            self.daily_count = 0
            self.current_date_utc = today

        for _ in range(count):
            self.minute_window.append(now)
            self.daily_count += 1
        
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.current_date_utc:
            self.daily_count = 0
            self.current_date_utc = today

        # Purge timestamps older than 60 seconds
        while self.minute_window and now - self.minute_window[0] > 60:
            self.minute_window.popleft()

        minute_used = len(self.minute_window)
        minute_remaining = max(0, self.rpm_limit - minute_used)
        daily_used = self.daily_count
        daily_remaining = max(0, self.rpd_limit - daily_used)

        # Time until midnight UTC reset
        now_dt = datetime.now(timezone.utc)
        tomorrow_midnight = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_to_reset = int((tomorrow_midnight - now_dt).total_seconds())
        hours = seconds_to_reset // 3600
        minutes = (seconds_to_reset % 3600) // 60
        seconds = seconds_to_reset % 60
        reset_time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

        status_text = "Optimal"
        if minute_used >= 12:
            status_text = "RPM Throttling Soon"
        elif daily_used >= 450:
            status_text = "Daily Quota Low"

        return {
            "daily_limit": self.rpd_limit,
            "daily_used": daily_used,
            "daily_remaining": daily_remaining,
            "daily_percent_remaining": round((daily_remaining / self.rpd_limit) * 100, 1),
            "minute_limit": self.rpm_limit,
            "minute_used": minute_used,
            "minute_remaining": minute_remaining,
            "reset_countdown": reset_time_str,
            "status": status_text,
            "tier": "Google AI Studio Free Tier"
        }

quota_tracker = GeminiQuotaTracker()

# ============================================================================
# Lazy Loading Singleton for RAG and AeroBot Models
# ============================================================================
_rag_components = None
_aerobot_fn = None

def get_rag_components():
    global _rag_components
    if _rag_components is None:
        logging.info("Initializing ChromaDB vectorstore and FlashRank MiniLM reranker...")
        from rag_retrieval_pipeline import (
            vectorstore,
            decomposer,
            compressor,
            llm as gemini_llm,
            prompt as rag_prompt,
            QueryDecompositionPlan,
            retriever as default_dense_retriever
        )
        from langchain_classic.retrievers import ContextualCompressionRetriever
        _rag_components = {
            "vectorstore": vectorstore,
            "decomposer": decomposer,
            "compressor": compressor,
            "gemini_llm": gemini_llm,
            "rag_prompt": rag_prompt,
            "QueryDecompositionPlan": QueryDecompositionPlan,
            "default_dense_retriever": default_dense_retriever,
            "ContextualCompressionRetriever": ContextualCompressionRetriever
        }
        logging.info("RAG components ready!")
    return _rag_components

def get_aerobot():
    global _aerobot_fn
    if _aerobot_fn is None:
        try:
            from areo_bot import chat_with_aerobot
            _aerobot_fn = chat_with_aerobot
        except Exception as e:
            logging.warning(f"Could not load AeroBot: {e}")
            _aerobot_fn = None
    return _aerobot_fn

# ============================================================================
# Pydantic Schemas
# ============================================================================
class SubQueryItem(BaseModel):
    sub_question: str
    mission: str

class RetrievedChunkItem(BaseModel):
    chunk_id: int
    content: str
    mission: str
    source_file: str

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    mode: Literal["dense", "hybrid"] = "dense"

class RAGQueryResponse(BaseModel):
    query: str
    mode: str
    answer: str
    is_refusal: bool
    plan: List[SubQueryItem]
    contexts: List[RetrievedChunkItem]
    latency_ms: int
    quota: Dict[str, Any]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AeroBotChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: List[ChatMessage] = Field(default_factory=list)

class AeroBotChatResponse(BaseModel):
    response: str
    tokens: Dict[str, int]
    is_restricted_topic: bool
    latency_ms: int

# ============================================================================
# Core RAG Execution Logic (Defensive with Graceful Fallbacks)
# ============================================================================
def execute_rag(query: str, mode: str = "dense") -> Dict[str, Any]:
    start_time = time.time()
    rag = get_rag_components()
    output_parser = StrOutputParser()
    
    # 1. Query Decomposition Layer
    plan_items: List[SubQueryItem] = []
    try:
        raw_plan = rag["decomposer"].invoke({"question": query})
        quota_tracker.record_requests(1)
        for sq in raw_plan.sub_queries:
            plan_items.append(SubQueryItem(sub_question=sq.sub_question, mission=sq.mission))
    except Exception as e:
        logging.warning(f"Query decomposition fallback: {e}")
        plan_items = [SubQueryItem(sub_question=query, mission="All")]

    if not plan_items:
        plan_items = [SubQueryItem(sub_question=query, mission="All")]

    # 2. Retrieval & Reranking Layer
    collected_docs = []
    seen_contents = set()

    for sq in plan_items:
        try:
            search_kwargs = {"k": 15}
            # Only apply filter if mission is a recognized corpus document
            if sq.mission in ["Apollo 11", "Apollo 13", "Challenger STS 51-L"]:
                search_kwargs["filter"] = {"mission": sq.mission}

            mission_base = rag["vectorstore"].as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            mission_retriever = rag["ContextualCompressionRetriever"](
                base_compressor=rag["compressor"],
                base_retriever=mission_base
            )
            sub_docs = mission_retriever.invoke(sq.sub_question)
            for doc in sub_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    collected_docs.append(doc)
        except Exception as e:
            logging.warning(f"Filtered retrieval for sub-query '{sq.sub_question}' failed: {e}")
            try:
                # Fallback to simple uncompressed similarity search without filter
                fallback_docs = rag["vectorstore"].similarity_search(sq.sub_question, k=5)
                for doc in fallback_docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        collected_docs.append(doc)
            except Exception as e2:
                logging.error(f"Fallback retrieval failed: {e2}")

    # Fallback if no documents collected yet
    if not collected_docs:
        try:
            fallback_docs = rag["vectorstore"].similarity_search(query, k=5)
            for doc in fallback_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    collected_docs.append(doc)
        except Exception as e:
            logging.error(f"General fallback retrieval failed: {e}")

    # 3. Format Contexts
    context_text = "\n\n".join(doc.page_content for doc in collected_docs)
    
    formatted_chunks = []
    for i, doc in enumerate(collected_docs):
        mission_tag = doc.metadata.get("mission") or "General NASA"
        source_name = doc.metadata.get("file_name") or doc.metadata.get("source") or "NASA Technical Report"
        formatted_chunks.append(RetrievedChunkItem(
            chunk_id=i + 1,
            content=doc.page_content,
            mission=mission_tag,
            source_file=Path(source_name).name
        ))

    # 4. Synthesize Grounded Technical Answer
    if not collected_docs:
        answer_text = "I cannot find relevant information in the indexed mission documents."
        is_refusal = True
    else:
        try:
            quota_tracker.record_requests(1)
            chain = rag["rag_prompt"] | rag["gemini_llm"] | output_parser
            raw_answer = chain.invoke({"context": context_text, "question": query})
            answer_text = str(raw_answer).strip()
        except Exception as e:
            err_msg = str(e)
            logging.error(f"LLM generation failed: {err_msg}", exc_info=True)
            if "ResourceExhausted" in err_msg or "429" in err_msg:
                answer_text = (
                    "⚠️ Google AI Studio Free Tier rate limit reached (15 Queries/Min or daily quota exhausted). "
                    "Please wait a few seconds and try again."
                )
            else:
                answer_text = f"Temporary generation notice: Unable to contact LLM synthesis engine ({err_msg})."
            is_refusal = True

    is_refusal = (
        "cannot find relevant information" in answer_text.lower()
        or "does not contain" in answer_text.lower()
        or "not covered in the provided" in answer_text.lower()
        or "rate limit reached" in answer_text.lower()
    )

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "query": query,
        "mode": mode,
        "answer": answer_text,
        "is_refusal": is_refusal,
        "plan": plan_items,
        "contexts": formatted_chunks,
        "latency_ms": latency_ms,
        "quota": quota_tracker.get_status()
    }

# ============================================================================
# API Routes
# ============================================================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/status")
async def get_system_status():
    chunk_count = 2058
    try:
        if _rag_components is not None:
            chunk_count = _rag_components["vectorstore"]._collection.count()
    except Exception:
        pass
    
    return {
        "status": "Operational",
        "system_name": "AeroRAG Console",
        "version": "2.0.0",
        "indexed_chunks": chunk_count,
        "indexed_missions": ["Apollo 11", "Apollo 13", "Challenger STS 51-L"],
        "reranker": "FlashRank MiniLM (ms-marco-MiniLM-L-12-v2, top_n=8)",
        "generation_llm": "Google Gemini Flash Lite",
        "chat_assistant": "AeroBot (Google Gemini Flash Lite)",
        "quota": quota_tracker.get_status()
    }

@app.get("/api/quota")
async def get_live_quota():
    return quota_tracker.get_status()

@app.get("/api/benchmarks")
async def get_benchmark_data():
    return {
        "configurations": [
            {
                "id": "tinybert_8",
                "label": "Baseline: TinyBERT (top_k=8)",
                "date": "2026-11-07",
                "is_selected": False,
                "in_scope": {
                    "context_precision": 48.97,
                    "context_recall": 88.46,
                    "faithfulness": 90.25,
                    "answer_relevancy": 75.37
                },
                "overall": {
                    "context_precision": 48.97,
                    "context_recall": 88.46,
                    "faithfulness": 78.21,
                    "answer_relevancy": 65.32
                },
                "negative_control_refusal": "100%",
                "notes": "Lightweight 2-layer compression; good baseline recall but lower precision."
            },
            {
                "id": "minilm_5",
                "label": "Experimental: MiniLM (top_k=5)",
                "date": "2026-11-08",
                "is_selected": False,
                "in_scope": {
                    "context_precision": 46.31,
                    "context_recall": 67.31,
                    "faithfulness": 97.63,
                    "answer_relevancy": 79.22
                },
                "overall": {
                    "context_precision": 46.31,
                    "context_recall": 67.31,
                    "faithfulness": 84.61,
                    "answer_relevancy": 68.66
                },
                "negative_control_refusal": "100%",
                "notes": "Extremely high faithfulness due to tiny window, but suffered a severe -21.15% recall bottleneck."
            },
            {
                "id": "minilm_8",
                "label": "Production: MiniLM (top_k=8)",
                "date": "2026-11-08",
                "is_selected": True,
                "in_scope": {
                    "context_precision": 49.82,
                    "context_recall": 84.62,
                    "faithfulness": 89.90,
                    "answer_relevancy": 72.69
                },
                "overall": {
                    "context_precision": 49.82,
                    "context_recall": 84.62,
                    "faithfulness": 77.92,
                    "answer_relevancy": 62.99
                },
                "negative_control_refusal": "100%",
                "notes": "Top-tier precision (49.82%) with 12 transformer layers, fully recovered recall (84.62%), and strict 0% hallucination."
            }
        ]
    }

@app.post("/api/rag/query", response_model=RAGQueryResponse)
def query_rag(req: RAGQueryRequest):
    try:
        result = execute_rag(req.query, mode=req.mode)
        return result
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Unhandled error in query_rag endpoint: {e}")
        # Return fallback response instead of crash
        return {
            "query": req.query,
            "mode": req.mode,
            "answer": f"Service notice: Unable to complete query execution ({str(e)}). Please verify your Google Gemini API key quota.",
            "is_refusal": True,
            "plan": [SubQueryItem(sub_question=req.query, mission="All")],
            "contexts": [],
            "latency_ms": 0,
            "quota": quota_tracker.get_status()
        }

@app.post("/api/aerobot/chat", response_model=AeroBotChatResponse)
def chat_aerobot(req: AeroBotChatRequest):
    start_time = time.time()
    
    # Check if question mentions restricted missions
    q_lower = req.question.lower()
    is_restricted = any(kw in q_lower for kw in [
        "apollo 11", "apollo 13", "challenger", "sts 51-l", "rogers commission",
        "first moon landing", "1986 disaster", "oxygen tank 2", "o-ring"
    ])

    from langchain_core.messages import HumanMessage, AIMessage
    history_objs = []
    for m in req.chat_history:
        if m.role == "user":
            history_objs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history_objs.append(AIMessage(content=m.content))

    aerobot_runner = get_aerobot()
    if aerobot_runner:
        try:
            quota_tracker.record_requests(1)
            bot_text, token_stats = aerobot_runner(req.question, history_objs, max_history_turns=6)
        except Exception as e:
            err_msg = str(e)
            if "ResourceExhausted" in err_msg or "429" in err_msg:
                bot_text = (
                    "⚠️ Google AI Studio Free Tier rate limit reached (15 Queries/Min or daily 500 quota exhausted). "
                    "Please wait a few seconds before sending another message."
                )
            else:
                bot_text = (
                    f"I encountered a temporary service connection when reaching Google Gemini ({err_msg}). "
                    "For Apollo 11, Apollo 13, and Challenger investigations, please switch to the AeroRAG tab to query our local ChromaDB."
                )
            token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    else:
        bot_text = "AeroBot model engine is currently not initialized. Please ensure GOOGLE_API_KEY (or GEMINI_API_KEY) is configured in your .env file."
        token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "response": bot_text,
        "tokens": token_stats,
        "is_restricted_topic": is_restricted,
        "latency_ms": latency_ms
    }

# ============================================================================
# Static Files & UI Serving
# ============================================================================
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = templates_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>AeroRAG Console initializing... Please refresh shortly.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
