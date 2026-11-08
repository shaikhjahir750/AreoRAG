from collections import defaultdict
import logging
import os
from pathlib import Path
import tempfile
from typing import List, Literal
from dotenv import load_dotenv

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

load_dotenv(override=True)

from flashrank import Ranker
from langchain_community.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "aerospace_mission_intel"

if not os.path.exists(PERSIST_DIR):
  raise FileNotFoundError(
      f"'{PERSIST_DIR}' does not exist. Run your ingestion script first!"
  )

# 1. Matching embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 2. Connect to ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR,
)

doc_count = vectorstore._collection.count()
print(f"Connected to ChromaDB! Total indexed chunks: {doc_count}")

# 3. Build global and per-mission BM25 corpora
raw_records = vectorstore.get()
all_docs = [
    Document(page_content=text, metadata=meta if meta else {})
    for text, meta in zip(raw_records["documents"], raw_records["metadatas"])
]

# Partition chunks by mission for filtered BM25 searches
mission_docs_map = defaultdict(list)
for doc in all_docs:
  m = doc.metadata.get("mission", "General NASA")
  mission_docs_map[m].append(doc)

# Global BM25 retriever
global_bm25 = BM25Retriever.from_documents(all_docs)
global_bm25.k = 10

# Per-mission BM25 retrievers
mission_bm25_map = {}
for m_tag, docs in mission_docs_map.items():
  retriever_inst = BM25Retriever.from_documents(docs)
  retriever_inst.k = 6
  mission_bm25_map[m_tag] = retriever_inst

# 4. Global FlashRank compressor
compressor = FlashrankRerank(
    client=Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir=tempfile.gettempdir()
    ), 
    top_n=8
)

# Global Fallback Hybrid Retriever (70% dense, 30% BM25)
global_dense = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 12}
)
global_ensemble = EnsembleRetriever(
    retrievers=[global_dense, global_bm25], weights=[0.7, 0.3]
)
global_hybrid_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=global_ensemble
)

# 5. LLM and Generation Prompt
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)

prompt = ChatPromptTemplate.from_template("""You are an expert NASA aerospace historian and technical investigator.
Use ONLY the provided technical context to answer the question.
If the answer cannot be directly derived from the context, respond strictly with:
"I cannot find relevant information in the indexed mission documents."

Keep your answer concise, fact-based, and under 100 words.

Context:
{context}

Question: {question}
Answer:""")


# 6. Structured Query Decomposition Schema
class SubQuery(BaseModel):
  """Data model for a single sub-query"""
  sub_question: str = Field(
      description=(
          "Self-contained search query focused exclusively on one mission"
      )
  )
  mission: Literal[
      "Apollo 11",
      "Apollo 13",
      "Challenger STS 51-L",
      "General NASA",
      "All",
  ] = Field(
      description=(
          "Target mission matching the indexed metadata: 'Apollo 11', 'Apollo"
          " 13', 'Challenger STS 51-L', 'General NASA', or 'All'"
      )
  )


class QueryDecompositionPlan(BaseModel):
  """Data model for the decomposition plan"""
  sub_queries: List[SubQuery] = Field(
      description=(
          "List of 1 to 3 distinct sub-queries targeting individual missions."
      )
  )


decompose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a NASA mission query planner. Decompose complex user"
            " questions comparing or inquiring about multiple aerospace"
            " missions into self-contained sub-queries targeting individual"
            " missions.\nAvailable mission tags: 'Apollo 11', 'Apollo 13',"
            " 'Challenger STS 51-L', 'General NASA'.\nIf the question asks"
            " about a single mission or a general topic, return exactly one"
            " sub-query with the matching mission or 'All'."
        ),
    ),
    ("human", "{question}"),
])

decomposer = decompose_prompt | llm.with_structured_output(
    QueryDecompositionPlan
)


# 7. Hybrid Context Retrieval with Prompt Decomposition
def retrieve_hybrid_decomposed_context(query: str) -> str:
  """Decomposes query, runs filtered hybrid searches per mission, and returns combined context."""
  try:
    plan: QueryDecompositionPlan = decomposer.invoke({"question": query})
  except Exception:
    docs = global_hybrid_retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

  # Single query with no mission filter -> use global hybrid retriever
  if len(plan.sub_queries) == 1 and plan.sub_queries[0].mission == "All":
    docs = global_hybrid_retriever.invoke(plan.sub_queries[0].sub_question)
    return "\n\n".join(d.page_content for d in docs)

  collected_docs = []
  seen_contents = set()

  for sq in plan.sub_queries:
    # 1. Mission-filtered dense retriever
    search_kwargs = {"k": 8}
    if sq.mission != "All":
      search_kwargs["filter"] = {"mission": sq.mission}

    sub_dense = vectorstore.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )

    # 2. Mission-filtered BM25 retriever
    sub_bm25 = mission_bm25_map.get(sq.mission, global_bm25)

    # 3. Sub-query Ensemble & Reranking
    sub_ensemble = EnsembleRetriever(
        retrievers=[sub_dense, sub_bm25], weights=[0.7, 0.3]
    )
    sub_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=sub_ensemble
    )

    # 4. Fetch and deduplicate
    docs = sub_retriever.invoke(sq.sub_question)
    for doc in docs:
      if doc.page_content not in seen_contents:
        seen_contents.add(doc.page_content)
        collected_docs.append(doc)

  if not collected_docs:
    return ""

  return "\n\n".join(doc.page_content for doc in collected_docs)


# 8. RAG Pipeline Chain
rag_chain = (
    {"context": retrieve_hybrid_decomposed_context, "question": lambda q: q}
    | prompt
    | llm
    | StrOutputParser()
)


def retrieve_docs(query: str):
  print(f"\nQuery: {query}")
  print("=" * 60)
  response = rag_chain.invoke(query)
  print("\nAnswer:\n", response)
  print("-" * 60)


if __name__ == "__main__":
  print(
      "Hybrid decomposed retriever ready! Type your question (or 'exit' to"
      " quit):"
  )
  while True:
    user_query = input("\nEnter query > ").strip()
    if not user_query:
      continue
    if user_query.lower() in ["exit", "quit", "q"]:
      break

    retrieve_docs(user_query)