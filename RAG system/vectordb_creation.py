"""
this script creates the croma db vector database 
if you run it multiple times it will eventually keep creating the copies of the existing 
emmbedings into croma db 
""" 
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Robust path resolution relative to repository root
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs" / "documents"
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "aerospace_mission_intel"

# Optimized batch parameters for Google Generative AI Free Tier
BATCH_SIZE = 25  # Pushes 25 chunks per API call (reduces total daily API calls by ~60%)
DELAY_BETWEEN_BATCHES = 8  # 8s delay keeps rate around 7.5 requests/min (well below 15 RPM limit)

# Helper to automatically assign structured metadata
def get_document_metadata(filename: str):
    name = filename.lower()
    agency = "NASA" if "nasa_" in name else "ISRO"

    if "apollo13" in name:
        mission, category = "Apollo 13", "Anomaly Investigation"
    elif "apollo11" in name:
        mission, category = "Apollo 11", "Lunar Landing Flight Report"
    elif "challenger" in name:
        mission, category = "Challenger STS 51-L", "Accident Board Investigation"
    elif "systems_handbook" in name:
        mission, category = "General NASA", "Systems Engineering Standards"
    elif "chandrayaan1" in name:
        mission, category = "Chandrayaan-1", "Lunar Orbit Insertion & Science"
    elif "aditya" in name:
        mission, category = "Aditya-L1", "Solar Heliophysics"
    elif "gaganyaan" in name:
        mission, category = "Gaganyaan TV-D1", "Crew Escape & In-Flight Abort"
    elif "achievements" in name:
        mission, category = "General ISRO", "Launch Vehicle & Technology Review"
    else:
        mission, category = "Unknown", "General Aerospace"

    return agency, mission, category

# Helper function to push batches with exponential backoff on 429
def ingest_batch_with_retry(vectorstore, batch_chunks, max_retries=3):
    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch_chunks)
            return True
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                wait_time = (attempt + 1) * 30
                print(f"      [!] Rate limit reached. Sleeping {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"      [!] Ingestion error on batch: {e}")
                raise e
    return False

def main():
    pdf_files = sorted(list(DOCS_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in '{DOCS_DIR}'. Run download script first.")
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

    # Initialize / open ChromaDB directly
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )

    for doc_idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n==================================================")
        print(f"Processing File [{doc_idx}/{len(pdf_files)}]: {pdf_path.name}")
        print(f"==================================================")

        # Check if file has already been ingested into ChromaDB
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

        agency, mission, category = get_document_metadata(pdf_path.name)

        # Enrich page metadata
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
            print(f"   Writing batch {current_batch_no}/{total_batches} ({len(batch)} chunks)...")
            
            success = ingest_batch_with_retry(vectorstore, batch)
            if not success:
                print(f"[!] Warning: Batch {current_batch_no} could not be ingested after retries.")
            time.sleep(DELAY_BETWEEN_BATCHES)

        print(f"✓ Completed indexing for: {pdf_path.name}")

    print(f"\nAll documents processed into '{PERSIST_DIR}' under collection '{COLLECTION_NAME}'!")

if __name__ == "__main__":
    main()