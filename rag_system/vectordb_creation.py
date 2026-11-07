"""
Aerospace Mission Intel - Ingestion Script for ChromaDB
Uses Google Generative AI Embeddings with explicit rate pacing.
"""
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs" / "documents"
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "aerospace_mission_intel"

# Safe Free-Tier Rate Constraints
BATCH_SIZE = 15            # 5 chunks (~1,250 tokens) avoids burst 429 rejections
DELAY_BETWEEN_BATCHES = 6.5  # Keeps throughput safely under 10 RPM (limit: 15 RPM)

def get_nasa_metadata(filename: str):
    """Fetches the documnets from the docs and  create the metadata for each document"""
    name = filename.lower()
    if "apollo13" in name or "apollo_13" in name:
        mission, category = "Apollo 13", "Anomaly Investigation"
    elif "apollo11" in name or "apollo_11" in name:
        mission, category = "Apollo 11", "Lunar Landing Flight Report"
    elif "challenger" in name:
        mission, category = "Challenger STS 51-L", "Accident Board Investigation"
    elif "systems_handbook" in name:
        mission, category = "General NASA", "Systems Engineering Standards"
    else:
        mission, category = "General NASA", "Aerospace Documentation"
    return "NASA", mission, category

def embed_and_upsert_batch(vectorstore, embedding_model, batch, batch_ids, max_retries=3):
    """Embedd chunks and upsert them into the vectorstore"""
    texts = [c.page_content for c in batch]
    metadatas = [c.metadata for c in batch]

    for attempt in range(max_retries):
        try:
            # 1. Direct sequential API call (avoids LangChain background thread bursts)
            vectors = embedding_model.embed_documents(texts)

            # 2. Direct Chroma write (no extra hidden API calls, upserts by ID)
            vectorstore._collection.upsert(
                ids=batch_ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                wait_time = (attempt + 1) * 35
                print(f"      [!] Rate limit reached. Backing off {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"      [✗] Error on batch: {e}")
                raise e
    return False

def main():
    """Main function to create the chromaDB"""
    pdf_files = sorted(list(DOCS_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in '{DOCS_DIR}'.")
        return

    print(f"Found {len(pdf_files)} PDF document(s) in {DOCS_DIR}:\n" + "\n".join(f"- {p.name}" for p in pdf_files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )

    for doc_idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n==================================================")
        print(f"Processing File [{doc_idx}/{len(pdf_files)}]: {pdf_path.name}")
        print(f"==================================================")

        # Skip document if already indexed
        existing = vectorstore.get(where={"file_name": pdf_path.name}, limit=1)
        if existing and len(existing.get("ids", [])) > 0:
            print(f"✓ Already indexed in ChromaDB. Skipping to save quota.")
            continue

        try:
            loader = PyPDFLoader(str(pdf_path))
            raw_pages = loader.load()
        except Exception as err:
            print(f"[✗] Failed reading {pdf_path.name}: {err}")
            continue

        agency, mission, category = get_nasa_metadata(pdf_path.name)
        clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem)

        for page in raw_pages:
            page.metadata["agency"] = agency
            page.metadata["mission"] = mission
            page.metadata["category"] = category
            page.metadata["file_name"] = pdf_path.name

        doc_chunks = splitter.split_documents(raw_pages)
        total_chunks = len(doc_chunks)
        print(f"-> Parsed {len(raw_pages)} pages | Created {total_chunks} chunks.")

        if total_chunks == 0:
            continue

        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = doc_chunks[i : i + BATCH_SIZE]
            current_batch_no = (i // BATCH_SIZE) + 1
            batch_ids = [f"{clean_stem}_chunk_{i + idx}" for idx in range(len(batch))]

            print(f"   Writing batch {current_batch_no}/{total_batches} ({len(batch)} chunks)...")
            success = embed_and_upsert_batch(vectorstore, embedding_model, batch, batch_ids)

            if not success:
                print(f"[!] Warning: Batch {current_batch_no} failed after retries.")

            time.sleep(DELAY_BETWEEN_BATCHES)

        print(f"✓ Completed indexing for: {pdf_path.name}")
        print("Cooling down 10s before next document...")
        time.sleep(10)

    print(f"\nAll documents successfully indexed into '{PERSIST_DIR}' [Collection: '{COLLECTION_NAME}']!")

if __name__ == "__main__":
    main()