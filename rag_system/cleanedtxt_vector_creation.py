"""Aerospace Mission Intel - Ingestion Script for Cleaned Challenger Data

Replaces corrupted Challenger PDF vectors with sanitized OCR text
into ChromaDB under collection 'aerospace_mission_intel'.
"""

import os
from pathlib import Path
import re
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# 1. Path & Collection Anchors
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs" / "documents"
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "aerospace_mission_intel"

# Safe Free-Tier Rate Constraints
BATCH_SIZE = 15
DELAY_BETWEEN_BATCHES = 6.5

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GOOGLE_API_KEY is missing from environment or .env file.")


def embed_and_upsert_batch(
    vectorstore, embedding_model, batch, batch_ids, max_retries=3
):
  """Embedds and upserts a batch of documents to the chromaDb"""

  texts = [c.page_content for c in batch]
  metadatas = [c.metadata for c in batch]

  for attempt in range(max_retries):
    try:
      # Sequential API call
      vectors = embedding_model.embed_documents(texts)

      # Direct Chroma write
      vectorstore._collection.upsert(
          ids=batch_ids,
          embeddings=vectors,
          documents=texts,
          metadatas=metadatas,
      )
      return True
    except Exception as e:
      err_msg = str(e)
      if (
          "429" in err_msg
          or "ResourceExhausted" in err_msg
          or "quota" in err_msg.lower()
      ):
        wait_time = (attempt + 1) * 35
        print(
            f"      [!] Rate limit reached. Backing off {wait_time}s (Attempt"
            f" {attempt + 1}/{max_retries})..."
        )
        time.sleep(wait_time)
      else:
        print(f"      [✗] Error on batch: {e}")
        raise e
  return False


def main():
  """Main function to create the chromaDB"""
  cleaned_txt_path = DOCS_DIR / "challenger_cleaned.txt"
  if not cleaned_txt_path.exists():
    print(f"Cleaned text file not found at: {cleaned_txt_path}")
    return

  print(
      f"Connecting to ChromaDB at '{PERSIST_DIR}' [Collection:"
      f" '{COLLECTION_NAME}']"
  )

  embedding_model = GoogleGenerativeAIEmbeddings(
      model="models/gemini-embedding-001",
      google_api_key=api_key,
  )

  vectorstore = Chroma(
      collection_name=COLLECTION_NAME,
      embedding_function=embedding_model,
      persist_directory=PERSIST_DIR,
  )

  # Purge old Challenger records to avoid duplicate/stale embeddings
  print(
      "\nPurging previous 'Challenger STS 51-L' chunks from collection to allow"
      " clean re-indexing..."
  )
  try:
    vectorstore.delete(where={"mission": "Challenger STS 51-L"})
    print("✓ Successfully cleared old Challenger vectors.")
  except Exception as e:
    print(f"Notice on purge: {e}")

  # Split cleaned text
  print(f"\nProcessing cleaned file: {cleaned_txt_path.name}")
  raw_text = cleaned_txt_path.read_text(encoding="utf-8")

  splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      separators=["\n\n", "\n", ". ", " ", ""],
  )

  # Match original metadata schema
  base_doc = Document(
      page_content=raw_text,
      metadata={
          "agency": "NASA",
          "mission": "Challenger STS 51-L",
          "category": "Accident Board Investigation",
          "file_name": "challenger_rogers_commission.pdf",
      },
  )

  doc_chunks = splitter.split_documents([base_doc])
  total_chunks = len(doc_chunks)
  print(f"-> Generated {total_chunks} chunks from cleaned corpus.")

  if total_chunks == 0:
    print("No chunks generated. Exiting.")
    return

  clean_stem = "challenger_rogers_commission"
  total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

  # Ingest with safe pacing
  for i in range(0, total_chunks, BATCH_SIZE):
    batch = doc_chunks[i : i + BATCH_SIZE]
    current_batch_no = (i // BATCH_SIZE) + 1
    batch_ids = [f"{clean_stem}_chunk_{i + idx}" for idx in range(len(batch))]

    print(
        f"   Writing batch {current_batch_no}/{total_batches} ({len(batch)}"
        " chunks)..."
    )
    success = embed_and_upsert_batch(
        vectorstore, embedding_model, batch, batch_ids
    )

    if not success:
      print(f"[!] Warning: Batch {current_batch_no} failed after retries.")

    if i + BATCH_SIZE < total_chunks:
      time.sleep(DELAY_BETWEEN_BATCHES)

  print(
      f"\n✓ Completed indexing cleaned Challenger documents into"
      f" '{COLLECTION_NAME}'!"
  )


if __name__ == "__main__":
  main()