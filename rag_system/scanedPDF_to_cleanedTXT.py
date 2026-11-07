"""clean_challenger_pdf.py

Extracts text from the NASA Challenger Rogers Commission report,
sanitizes corrupted OCR artifacts (e.g., 'en$neering' -> 'engineering'),
and prepares clean documents for vectorstore indexing.
"""

from pathlib import Path
import re
import fitz  # PyMuPDF
import ftfy
from langchain_core.documents import Document


def clean_ocr_text(text: str) -> str:
  """Sanitizes text extracted from scanned PDF OCR artifacts."""
  # 1. Fix broken unicode encoding / mojibake
  text = ftfy.fix_text(text)

  # 2. Fix hyphenated line wraps (e.g., "engi-\nneering" -> "engineering")
  text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

  # 3. Fix OCR special-character substitutions inside alphabetic words
  ocr_char_map = [
      (
          r"(\b[a-zA-Z]+)\$([a-zA-Z]+\b)",
          r"\1gi\2",
      ),  # en$neering -> engineering
      (r"(\b[a-zA-Z]+)0([a-zA-Z]+\b)", r"\1o\2"),  # c0mpany -> company
      (r"(\b[a-zA-Z]+)1([a-zA-Z]+\b)", r"\1l\2"),  # f1ight -> flight
      (r"(\b[a-zA-Z]+)@([a-zA-Z]+\b)", r"\1a\2"),  # d@ta -> data
      (r"(\b[a-zA-Z]+)5([a-zA-Z]+\b)", r"\1s\2"),  # mi55ion -> mission
  ]
  for pattern, repl in ocr_char_map:
    text = re.sub(pattern, repl, text)

  # 4. Remove unprintable ASCII control characters (retain newlines and tabs)
  text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

  # 5. Remove isolated stray scanner noise symbols surrounded by spaces
  text = re.sub(r"(?<=\s)[~`^|\\_§©®™<>](?=\s)", "", text)

  # 6. Normalize multiple horizontal spaces and redundant blank lines
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r"\n\s*\n+", "\n\n", text)

  return text.strip()


def extract_challenger_pdf(
    pdf_path: str | Path,
    output_txt_path: str | Path = None,
    min_page_chars: int = 50,
) -> list[Document]:
  """Extracts page text from the Challenger PDF using PyMuPDF, cleans OCR noise,

  and returns a list of LangChain Document objects ready for chunking.
  """
  pdf_file = Path(pdf_path)
  if not pdf_file.exists():
    raise FileNotFoundError(f"Challenger PDF not found at: {pdf_file.resolve()}")

  doc = fitz.open(pdf_file)
  documents: list[Document] = []
  corpus_chunks: list[str] = []

  print(f"Loading '{pdf_file.name}' ({len(doc)} pages)...")

  for page_index in range(len(doc)):
    page = doc[page_index]
    page_number = page_index + 1

    # Extract text preserving spatial layout
    raw_text = page.get_text("text")
    cleaned_text = clean_ocr_text(raw_text)

    # Skip blank or low-information pages (e.g., scanned blanks or dividers)
    if len(cleaned_text) < min_page_chars:
      continue

    # Metadata matching the AeroRAG multi-document filter format
    metadata = {
        "source": pdf_file.name,
        "mission": "challenger",
        "page": page_number,
    }

    documents.append(Document(page_content=cleaned_text, metadata=metadata))
    corpus_chunks.append(f"=== PAGE {page_number} ===\n{cleaned_text}\n")

  print(
      f"Extracted and cleaned {len(documents)} valid pages out of {len(doc)}."
  )

  # Optional: write out flat cleaned text file for manual spot-checking
  if output_txt_path:
    out_file = Path(output_txt_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(corpus_chunks), encoding="utf-8")
    print(f"Exported plain text review file to: {out_file.resolve()}")

  return documents


if __name__ == "__main__":
  input_pdf = Path(
      r"C:\Users\shaik\OneDrive\Documents\GitHub\AreoRAG\docs\documents\nasa_challenger_rogers_commission.pdf"
  )
  clean_export_path = Path(
      r"C:\Users\shaik\OneDrive\Documents\GitHub\AreoRAG\docs\documents\challenger_cleaned.txt"
  )

  cleaned_docs = extract_challenger_pdf(
      pdf_path=input_pdf, output_txt_path=clean_export_path
  )

  if cleaned_docs:
    print("\n--- SAMPLE EXTRACT (Page 1 Metadata & First 400 Characters) ---")
    print("Metadata:", cleaned_docs[0].metadata)
    print("Snippet:\n", cleaned_docs[0].page_content[:400], "...\n")