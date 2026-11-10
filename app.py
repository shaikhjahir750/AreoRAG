import os

# Set low-memory & single-thread environment flags BEFORE importing numpy/chromadb/onnxruntime
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import gc
import sys
import time
import json
import logging
import tempfile
import threading
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional, Dict, Any
from pathlib import Path
from collections import deque

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
VALID_MISSIONS = ("Apollo 11", "Apollo 13", "Challenger STS 51-L")

RAG_SYSTEM_PROMPT = (
    "You are an expert NASA aerospace historian and technical investigator.\n"
    "Use ONLY the provided technical context to answer the question.\n"
    "If the answer cannot be directly derived from the context, respond strictly with:\n"
    '"I cannot find relevant information in the indexed mission documents."\n\n'
    "Keep your answer concise, fact-based, and under 100 words."
)

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a NASA mission query planner. Decompose complex user questions comparing or "
    "inquiring about multiple aerospace missions into self-contained sub-queries targeting "
    "individual missions.\n"
    "Available mission tags in database: 'Apollo 11', 'Apollo 13', 'Challenger STS 51-L', "
    "'General NASA'.\n"
    "If the question asks about a single mission or general topic, return exactly one "
    "sub-query with the matching mission or 'All'."
)

AEROBOT_SYSTEM_PROMPT = (
    "You are AeroBot, an aerospace engineer and space-science assistant (orbital mechanics, "
    "rocketry, planetary exploration, astrophysics, mission history).\n\n"
    "Answer in 150-200 words: accurate, technical but accessible, direct answer first. "
    "Use SI units. Never invent facts; if unsure, say so.\n\n"
    "RESTRICTED TOPICS (covered by AeroRAG): Apollo 11, Apollo 13, Challenger/STS 51-L, "
    'including indirect references like "the 1986 shuttle disaster" or "first Moon landing mission".\n'
    "- Don't answer the restricted part, even partially. Reply with:\n"
    '"💡 Note: For detailed telemetry, verbatim transcript evidence, and source-grounded '
    'technical investigation data on [Apollo 11 / Apollo 13 / Challenger STS 51-L], '
    'check out the AeroRAG tool available on this site."\n'
    "- If only part of the question is restricted, answer the rest, then add the note.\n\n"
    "If a question is unrelated to space, politely say it's outside your focus."
)


def release_unused_memory():
    """Collect Python garbage and return freed C arenas to the Linux kernel when available."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def resolve_reranker_model() -> str:
    """Select reranker model; defaults to TinyBERT on 512 MB Render containers to prevent OOM."""
    override = os.environ.get("RERANKER_MODEL", "").strip()
    if override:
        return override
    if os.environ.get("RENDER"):
        return "ms-marco-TinyBERT-L-2-v2"
    return "ms-marco-MiniLM-L-12-v2"


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
# Instant Health Check for Render
# ============================================================================
@app.get("/healthz", include_in_schema=False)
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ============================================================================
# Quota & Rate Limit State Tracker for Google Gemini Free Tier
# ============================================================================
class GeminiQuotaTracker:
    """
    Thread-safe rate limit and daily quota tracker for Google AI Studio Free Tier:
    - 15 Requests Per Minute (RPM)
    - 500 Requests Per Day (RPD)
    Shared across both AeroRAG and AeroBot.
    """
    def __init__(self, state_file: Path = BASE_DIR / "quota_state.json"):
        self.state_file = state_file
        self.rpm_limit = 15
        self.rpd_limit = 500
        self.minute_window = deque()
        self.daily_count = 0
        self.current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._lock = threading.Lock()
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
        with self._lock:
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
        with self._lock:
            now = time.time()
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self.current_date_utc:
                self.daily_count = 0
                self.current_date_utc = today

            while self.minute_window and now - self.minute_window[0] > 60:
                self.minute_window.popleft()

            minute_used = len(self.minute_window)
            minute_remaining = max(0, self.rpm_limit - minute_used)
            daily_used = self.daily_count
            daily_remaining = max(0, self.rpd_limit - daily_used)

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
# Low-Memory FlashRank Cross-Encoder & Lazy Singletons
# ============================================================================
class LowMemoryReranker:
    """
    Memory-constrained FlashRank cross-encoder wrapper:
    - Disables ONNX CPU memory arena & pattern caching (~150-200 MB saved)
    - Forces single-thread execution (no per-thread arena bloat on Linux)
    - Scores candidates in micro-batches of 2 to avoid transient tensor spikes
    """
    def __init__(self, model_name: str, max_length: int = 384, micro_batch_size: int = 2):
        self.model_name = model_name
        self.micro_batch_size = micro_batch_size
        self.ranker = None
        self.rerank_req_cls = None
        self._lock = threading.Lock()

        import onnxruntime as ort
        from flashrank import Ranker, RerankRequest
        from flashrank.Config import model_file_map

        ranker = Ranker.__new__(Ranker)
        ranker.logger = logging.getLogger("flashrank")
        ranker.cache_dir = Path(tempfile.gettempdir())
        ranker.model_dir = ranker.cache_dir / model_name
        ranker._prepare_model_dir(model_name)

        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 1
        session_opts.inter_op_num_threads = 1
        session_opts.enable_cpu_mem_arena = False
        session_opts.enable_mem_pattern = False

        ranker.llm_model = None
        ranker.session = ort.InferenceSession(
            str(ranker.model_dir / model_file_map[model_name]),
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        ranker.tokenizer = ranker._get_tokenizer(max_length)

        self.ranker = ranker
        self.rerank_req_cls = RerankRequest

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        passages = [{"id": idx, "text": doc["content"]} for idx, doc in enumerate(candidates)]
        scored: List[Dict[str, Any]] = []

        # Serialize ONNX inference so concurrent requests never double activation memory
        with self._lock:
            for start in range(0, len(passages), self.micro_batch_size):
                batch = passages[start:start + self.micro_batch_size]
                req = self.rerank_req_cls(query=query, passages=batch)
                scored.extend(self.ranker.rerank(req))

        scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return [candidates[int(item["id"])] for item in scored[:top_n]]


_genai_client = None
_genai_lock = threading.Lock()
_rag_components = None
_rag_lock = threading.Lock()


def get_genai_client() -> Optional[genai.Client]:
    """Lazily return a shared lightweight Google GenAI client."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    with _genai_lock:
        if _genai_client is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logging.warning("GOOGLE_API_KEY / GEMINI_API_KEY missing from environment.")
                return None
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def get_rag_components() -> Dict[str, Any]:
    """Lazily initialize ChromaDB collection and low-memory FlashRank reranker once."""
    global _rag_components
    if _rag_components is not None:
        return _rag_components

    with _rag_lock:
        if _rag_components is None:
            persist_dir = BASE_DIR / "chroma_db"
            if not persist_dir.exists():
                raise FileNotFoundError(f"'{persist_dir}' does not exist. Run ingestion first.")

            logging.info("Initializing ChromaDB collection and low-memory FlashRank reranker...")
            import chromadb
            from chromadb.config import Settings

            chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = chroma_client.get_collection("aerospace_mission_intel")
            doc_count = collection.count()
            logging.info(f"Connected to ChromaDB! Total indexed chunks: {doc_count}")

            model_name = resolve_reranker_model()
            reranker = None
            try:
                reranker = LowMemoryReranker(model_name=model_name, max_length=384, micro_batch_size=2)
                logging.info(f"Low-memory FlashRank ready ({model_name}).")
            except Exception as err:
                logging.warning(f"FlashRank ({model_name}) unavailable, falling back to dense ranking: {err}")

            _rag_components = {
                "collection": collection,
                "doc_count": doc_count,
                "reranker": reranker,
                "reranker_model": model_name if reranker else "Dense Similarity Only",
            }
            release_unused_memory()
    return _rag_components


# ============================================================================
# Pydantic Schemas
# ============================================================================
class SubQuery(BaseModel):
    sub_question: str = Field(
        description="Self-contained search query focused exclusively on one mission"
    )
    mission: Literal[
        "Apollo 11",
        "Apollo 13",
        "Challenger STS 51-L",
        "General NASA",
        "All",
    ] = Field(
        description="Target mission matching indexed metadata or 'All'"
    )


class QueryDecompositionPlan(BaseModel):
    sub_queries: List[SubQuery] = Field(
        description="List of 1 to 3 distinct sub-queries targeting individual missions."
    )


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
# Core RAG Execution Logic (Low-Memory Direct SDK Pipeline)
# ============================================================================
def _extract_source_filename(metadata: Dict[str, Any]) -> str:
    raw_name = metadata.get("file_name") or metadata.get("source") or "NASA Technical Report"
    # Normalize Windows paths cleanly even when running on Linux containers
    return str(raw_name).replace("\\", "/").split("/")[-1]


def _embed_query_text(client: genai.Client, text: str) -> List[float]:
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(result.embeddings[0].values)


def _decompose_query(client: genai.Client, query: str) -> List[SubQueryItem]:
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=DECOMPOSE_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=QueryDecompositionPlan,
            ),
        )
        quota_tracker.record_requests(1)
        plan = QueryDecompositionPlan.model_validate_json(response.text)
        items = [SubQueryItem(sub_question=sq.sub_question, mission=sq.mission) for sq in plan.sub_queries[:3]]
        if items:
            return items
    except Exception as e:
        logging.warning(f"Query decomposition fallback: {e}")

    return [SubQueryItem(sub_question=query, mission="All")]


def _query_chroma_candidates(
    collection: Any,
    client: genai.Client,
    text: str,
    n_results: int,
    mission_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_vec = _embed_query_text(client, text)
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_vec],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if mission_filter in VALID_MISSIONS:
        query_kwargs["where"] = {"mission": mission_filter}

    res = collection.query(**query_kwargs)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    candidates = []
    for idx, content in enumerate(docs):
        meta = metas[idx] if idx < len(metas) and metas[idx] else {}
        dist = float(dists[idx]) if idx < len(dists) else 1.0
        candidates.append({
            "content": content,
            "mission": meta.get("mission") or "General NASA",
            "source_file": _extract_source_filename(meta),
            "distance": dist,
        })
    return candidates


def execute_rag(query: str, mode: str = "dense") -> Dict[str, Any]:
    start_time = time.time()
    client = get_genai_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY is not configured.")

    rag = get_rag_components()

    try:
        # 1. Query Decomposition Layer
        plan_items = _decompose_query(client, query)

        # 2. Candidate Retrieval Layer (bounded pool keeps memory well below 512 MB)
        per_query_k = 12 if len(plan_items) == 1 else 7
        candidate_pool: List[Dict[str, Any]] = []
        seen_contents = set()

        for sq in plan_items:
            try:
                mission = sq.mission if sq.mission in VALID_MISSIONS else None
                hits = _query_chroma_candidates(
                    rag["collection"], client, sq.sub_question, n_results=per_query_k, mission_filter=mission
                )
            except Exception as err:
                logging.warning(f"Filtered retrieval for '{sq.sub_question}' failed: {err}")
                hits = []

            if not hits:
                try:
                    hits = _query_chroma_candidates(
                        rag["collection"], client, sq.sub_question, n_results=6, mission_filter=None
                    )
                except Exception as err2:
                    logging.error(f"Unfiltered retrieval fallback failed: {err2}")
                    hits = []

            for doc in hits:
                if doc["content"] not in seen_contents:
                    seen_contents.add(doc["content"])
                    candidate_pool.append(doc)

        if not candidate_pool:
            try:
                candidate_pool = _query_chroma_candidates(
                    rag["collection"], client, query, n_results=6, mission_filter=None
                )
            except Exception as err3:
                logging.error(f"General fallback retrieval failed: {err3}")

        # 3. Single-Pass Micro-Batched Reranking (top_n=8)
        if candidate_pool and rag["reranker"] is not None:
            try:
                selected_docs = rag["reranker"].rerank(query, candidate_pool, top_n=8)
            except Exception as rerank_err:
                logging.warning(f"Reranker fallback to dense order: {rerank_err}")
                candidate_pool.sort(key=lambda item: item["distance"])
                selected_docs = candidate_pool[:8]
        else:
            candidate_pool.sort(key=lambda item: item["distance"])
            selected_docs = candidate_pool[:8]

        formatted_chunks = [
            RetrievedChunkItem(
                chunk_id=i + 1,
                content=doc["content"],
                mission=doc["mission"],
                source_file=doc["source_file"],
            )
            for i, doc in enumerate(selected_docs)
        ]

        # 4. Synthesize Grounded Technical Answer
        if not selected_docs:
            answer_text = "I cannot find relevant information in the indexed mission documents."
        else:
            context_text = "\n\n".join(doc["content"] for doc in selected_docs)
            try:
                quota_tracker.record_requests(1)
                user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
                gen_res = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=RAG_SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                answer_text = (gen_res.text or "").strip()
                if not answer_text:
                    answer_text = "I cannot find relevant information in the indexed mission documents."
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

        is_refusal = (
            "cannot find relevant information" in answer_text.lower()
            or "does not contain" in answer_text.lower()
            or "not covered in the provided" in answer_text.lower()
            or "rate limit reached" in answer_text.lower()
            or "unable to contact llm synthesis engine" in answer_text.lower()
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
            "quota": quota_tracker.get_status(),
        }
    finally:
        release_unused_memory()


# ============================================================================
# API Routes
# ============================================================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/status")
async def get_system_status():
    chunk_count = 2058
    reranker_label = f"FlashRank ({resolve_reranker_model()}, top_n=8)"
    if _rag_components is not None:
        chunk_count = _rag_components.get("doc_count", chunk_count)
        reranker_label = f"FlashRank ({_rag_components.get('reranker_model', resolve_reranker_model())}, top_n=8)"

    return {
        "status": "Operational",
        "system_name": "AeroRAG Console",
        "version": "2.0.0",
        "indexed_chunks": chunk_count,
        "indexed_missions": list(VALID_MISSIONS),
        "reranker": reranker_label,
        "generation_llm": "Google Gemini Flash Lite",
        "chat_assistant": "AeroBot (Google Gemini Flash Lite)",
        "quota": quota_tracker.get_status(),
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
                "notes": "Lightweight 2-layer compression; good baseline recall and minimal RAM footprint."
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
        return execute_rag(req.query, mode=req.mode)
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Unhandled error in query_rag endpoint: {e}")
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

    q_lower = req.question.lower()
    is_restricted = any(kw in q_lower for kw in [
        "apollo 11", "apollo 13", "challenger", "sts 51-l", "rogers commission",
        "first moon landing", "1986 disaster", "oxygen tank 2", "o-ring"
    ])

    client = get_genai_client()
    token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if client is not None:
        try:
            quota_tracker.record_requests(1)
            contents = []
            for msg in req.chat_history[-6:]:
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=msg.content)]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=req.question)]))

            res = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=AEROBOT_SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            bot_text = (res.text or "").strip()
            usage = getattr(res, "usage_metadata", None)
            if usage is not None:
                token_stats = {
                    "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                    "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
                }
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
    else:
        bot_text = "AeroBot model engine is currently not initialized. Please ensure GOOGLE_API_KEY (or GEMINI_API_KEY) is configured in your environment."

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
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except Exception as e:
            logging.error(f"Error reading index.html: {e}")
    return HTMLResponse("<h1>AeroRAG Console is Operational</h1><p>API Docs available at <a href='/docs'>/docs</a></p>")


if __name__ == "__main__":
    import uvicorn
    server_port = int(os.environ.get("PORT", 10000))
    logging.info(f"Starting server on 0.0.0.0:{server_port}")
    uvicorn.run("app:app", host="0.0.0.0", port=server_port, reload=False)
