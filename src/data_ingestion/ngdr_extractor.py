"""
NGDR (National Geoscience Data Repository) Automated Ingestion & Extraction Engine.
Automatically unzips and ingests shapefiles, geochemical tables, and PDFs
downloaded from geodataindia.gov.in for Katghora, Korba District.
"""

import os
import glob
import zipfile
import shutil
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DOWNLOADS_DIR = os.path.expanduser('~/Downloads')

def ingest_ngdr_downloads(source_dir=None):
    if source_dir is None:
        source_dir = DOWNLOADS_DIR
    
    zip_files = glob.glob(os.path.join(source_dir, '*geodata*.zip')) +                 glob.glob(os.path.join(source_dir, '*ngdr*.zip')) +                 glob.glob(os.path.join(source_dir, '*korba*.zip')) +                 glob.glob(os.path.join(source_dir, '*katghora*.zip')) +                 glob.glob(os.path.join(DATA_DIR, '*.zip'))
    
    if not zip_files:
        print(f'[NGDR Ingestion] No pending zip archives found in {source_dir} or {DATA_DIR}.')
        print('[NGDR Ingestion] Drop your downloaded NGDR zip file into Downloads or data/ and run again!')
        return False
        
    for zf in zip_files:
        print(f'[NGDR Ingestion] Found NGDR archive: {zf}')
        extract_to = os.path.join(DATA_DIR, 'ngdr_raw', os.path.splitext(os.path.basename(zf))[0])
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(extract_to)
        print(f'[NGDR Ingestion] Successfully extracted to: {extract_to}')
    return True

if __name__ == '__main__':
    ingest_ngdr_downloads()
