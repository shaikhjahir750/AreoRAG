"""
This script retrieves information from a local ChromaDB database of aerospace mission documents.
It uses similarity search to retrieve relavant chunks from the database.
It uses a decomposition strategy to break down complex queries into smaller sub-queries, 
which are then used to retrieve relevant documents from the database.
"""

import os
from pathlib import Path
import tempfile
from typing import List, Literal
from dotenv import load_dotenv
from flashrank import Ranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv(override=True)

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

# 2. Connect using the exact collection name
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR,
)

# Quick sanity check: Verify database has records
doc_count = vectorstore._collection.count()
print(f"Connected to ChromaDB! Total indexed chunks: {doc_count}")

# 3. Base Retriever & FlashRank Compression
class SafeFlashrankRerank(FlashrankRerank):
    """Wraps FlashrankRerank to safely handle empty document lists without ONNX input errors."""
    def compress_documents(self, documents, query, callbacks=None):
        if not documents:
            return []
        return super().compress_documents(documents, query, callbacks=callbacks)

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15},
)

compressor = SafeFlashrankRerank(
    client=Ranker(
        model_name="ms-marco-MiniLM-L-12-v2", 
        cache_dir=tempfile.gettempdir()
    ),
    top_n=8
)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# 4. LLM & Primary Generation Prompt
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


# ==========================================
# 5. Prompt Decomposition Architecture
# ==========================================
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
            " missions.\nAvailable mission tags in database: 'Apollo 11',"
            " 'Apollo 13', 'Challenger STS 51-L', 'General NASA'.\nIf the"
            " question asks about a single mission or general topic, return"
            " exactly one sub-query with the matching mission or 'All'."
        ),
    ),
    ("human", "{question}"),
])

decomposer = decompose_prompt | llm.with_structured_output(
    QueryDecompositionPlan
)


def retrieve_decomposed_context(query: str) -> str:
  """Decomposes query, runs filtered searches per mission, and returns combined deduplicated context."""
  try:
    plan: QueryDecompositionPlan = decomposer.invoke({"question": query})
  except Exception as e:
    # Fallback to standard retriever if decomposition fails
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

  # If only 1 sub-query with no specific mission filter, use base retriever + FlashRank
  if len(plan.sub_queries) == 1 and plan.sub_queries[0].mission == "All":
    docs = retriever.invoke(plan.sub_queries[0].sub_question)
    return "\n\n".join(d.page_content for d in docs)

  collected_docs = []
  seen_contents = set()

  for sq in plan.sub_queries:
    search_kwargs = {"k": 15}
    if sq.mission != "All":
      search_kwargs["filter"] = {"mission": sq.mission}

    mission_base_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )
    mission_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=mission_base_retriever
    )

    docs = mission_retriever.invoke(sq.sub_question)
    for doc in docs:
      if doc.page_content not in seen_contents:
        seen_contents.add(doc.page_content)
        collected_docs.append(doc)

  if not collected_docs:
    return ""

  return "\n\n".join(doc.page_content for doc in collected_docs)


# 6. Final Pipeline Chain
rag_chain = (
    {"context": retrieve_decomposed_context, "question": lambda q: q}
    | prompt
    | llm
    | StrOutputParser()
)


# 7. Retrieval function
def retrieve_docs(query: str):
  print(f"\nQuery: {query}")
  print("=" * 60)

  # Invoke the pipeline directly
  response = rag_chain.invoke(query)

  print("\nAnswer:\n", response)
  print("-" * 60)


if __name__ == "__main__":
  print("Retriever ready! Type your question (or 'exit' to quit):")
  while True:
    user_query = input("\nEnter query > ").strip()
    if not user_query:
      continue
    if user_query.lower() in ["exit", "quit", "q"]:
      break

    retrieve_docs(user_query)