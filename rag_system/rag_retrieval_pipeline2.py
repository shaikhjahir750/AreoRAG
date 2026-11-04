import os
import tempfile
import logging
from dotenv import load_dotenv

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

load_dotenv(override=True)

from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "aerospace_mission_intel"

if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(
        f"'{PERSIST_DIR}' does not exist. Run your ingestion script first!"
    )

# 1. Matching embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 2. Connect to ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR,
)

doc_count = vectorstore._collection.count()
print(f"Connected to ChromaDB! Total indexed chunks: {doc_count}")

# 3. Dense Retriever (Semantic Vector Search)
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# 4. Sparse Retriever (BM25 Keyword Search)
# Extract indexed documents directly from ChromaDB to populate BM25
raw_records = vectorstore.get()
all_docs = [
    Document(
        page_content=text,
        metadata=meta if meta else {}
    )
    for text, meta in zip(raw_records["documents"], raw_records["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 6

# 5. Hybrid Ensemble Retriever (RRF blending: 50% dense, 50% sparse)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# 6. Reranking Layer (FlashRank applied on top of the ensemble results)
compressor = FlashrankRerank(
    client=Ranker(cache_dir=tempfile.gettempdir()),
    top_n=3
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# 7. LLM, Prompt, and Pipeline
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)

prompt = ChatPromptTemplate.from_template(
    """You are an expert NASA aerospace historian and technical investigator.
Use ONLY the provided technical context to answer the question.
If the answer cannot be directly derived from the context, respond strictly with:
"I cannot find relevant information in the indexed mission documents."

Keep your answer concise, fact-based, and under 100 words.

Context:
{context}

Question: {question}
Answer:"""
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
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
    print("Hybrid retriever ready! Type your question (or 'exit' to quit):")
    while True:
        user_query = input("\nEnter query > ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            break

        retrieve_docs(user_query)