"""
this script is to download the open source documents that required for the rag project.
these document are stored in the docs folder as raw data.
these pdfs are chunked and stored in the chromadb vector database.
"""

import os
import requests
from pathlib import Path

PDF_DIR = Path("documents")
PDF_DIR.mkdir(parents=True, exist_ok=True)

DOCS = {
    # 4 NASA Documents
    "nasa_apollo13_review.pdf": "https://ntrs.nasa.gov/api/citations/20170000913/downloads/20170000913.pdf",
    "nasa_challenger_rogers_commission.pdf": "https://ntrs.nasa.gov/api/citations/19860015255/downloads/19860015255.pdf",
    "nasa_apollo11_mission_report.pdf": "https://ntrs.nasa.gov/api/citations/19710015566/downloads/19710015566.pdf",
    "nasa_systems_engineering_handbook.pdf": "https://ntrs.nasa.gov/api/citations/20170007230/downloads/20170007230.pdf",
    
    # 4 ISRO Documents
     "isro_aditya_l1_booklet.pdf": "https://www.isro.gov.in/media_isro/pdf/Aditya_L1_second_booklet.pdf",
    "isro_gaganyaan_tvd1.pdf": "https://www.isro.gov.in/media_isro/pdf/TVD1/TVD1-Brochure.pdf",
    "isro_space_achievements_tech.pdf": "https://www.isro.gov.in/media_isro/pdf/9Years_Space_Achievement_Feb2024.pdf",
    "isro_chandrayaan1_publication.pdf": "https://www.isro.gov.in/media_isro/pdf/chandrayaan/PSLV_C11_PUBLICATION.pdf",
}


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

for filename, url in DOCS.items():
    dest = PDF_DIR / filename
    if not dest.exists():
        print(f"Downloading {filename}...")
        try:
            res = requests.get(url, headers=headers, stream=True, timeout=90)
            res.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in res.iter_content(chunk_size=16384):
                    f.write(chunk)
            print(f"✓ Saved: {filename} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"✗ Failed {filename}: {e}")
    else:
        print(f"- Already present: {filename}")