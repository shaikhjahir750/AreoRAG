import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

try:
    model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    res = model.embed_query("test query")
    print("SUCCESS: Your API key is active. Daily limit is NOT exceeded.")
except Exception as e:
    print("FAILURE DETAILS:")
    print(e)
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb

# Point directly to your AeroRAG project root
BASE_DIR = Path(r"C:\Users\shaik\OneDrive\Documents\GitHub\AreoRAG")
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "aerospace_mission_intel"

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# Check existing chunks for this specific file
target_file = "nasa_challenger_rogers_commission.pdf"
results = collection.get(where={"file_name": target_file})
found_ids = results.get("ids", [])

print(f"Found {len(found_ids)} existing chunk(s) for '{target_file}'.")

if found_ids:
    collection.delete(ids=found_ids)
    print(f"✓ Successfully deleted all {len(found_ids)} chunks for '{target_file}'.")
else:
    print("No chunks found with that exact file_name. Checking all IDs in collection...")
    all_data = collection.get()
    matching_ids = [
        _id for _id in all_data.get("ids", [])
        if "nasa_challenger_rogers_commission" in _id
    ]
    if matching_ids:
        collection.delete(ids=matching_ids)
        print(f"✓ Deleted {len(matching_ids)} chunks by ID matching.")
    else:
        print("Collection is already clear of this file.")

