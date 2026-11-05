import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
import tempfile
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank

load_dotenv(override=True)

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

# 2. Connect using the exact collection name
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR,
)

# Quick sanity check: Verify database has records
doc_count = vectorstore._collection.count()
print(f"Connected to ChromaDB! Total indexed chunks: {doc_count}")

# 3. Create retriever with standard similarity search (no threshold dropping)
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 15, 
    }
)

compressor = FlashrankRerank(client=Ranker(cache_dir=tempfile.gettempdir()), top_n=8)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. LLM, Prompt, and Pipeline using the pipe operator (|)
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
# this chain retrieves documents, formats prompt, queries LLM, and parses output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Retrieval function
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