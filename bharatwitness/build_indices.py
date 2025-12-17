#!/usr/bin/env python3
"""
Quick script to build indices from the corpus data
"""

import json
import logging
from pathlib import Path
from pipeline.chunker import TextChunk
from pipeline.index_build import HybridIndexBuilder
from utils.logging_utils import setup_logging
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_chunks_from_corpus(corpus_dir: Path, max_files: int = 50) -> list:
    """Extract text chunks from the corpus JSON files"""
    chunks = []
    
    json_files = list(corpus_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in corpus")
    
    # Limit to max_files for faster indexing during development
    json_files = json_files[:max_files]
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract category and search results
            category = data.get('category', 'Unknown Category')
            search_results = data.get('search_results', [])
            
            # Create a document text from the category and search results
            doc_text = f"Category: {category}\n\n"
            
            for result in search_results[:5]:  # Limit to first 5 results per file
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                if title or snippet:
                    doc_text += f"{title}\n{snippet}\n\n"
            
            # Create a chunk from this document
            chunk_id = f"chunk_{idx}_{hashlib.md5(json_file.name.encode()).hexdigest()[:8]}"
            
            chunk = TextChunk(
                chunk_id=chunk_id,
                text=doc_text.strip(),
                byte_start=0,
                byte_end=len(doc_text.encode('utf-8')),
                page_nums=[0],
                section_types=['general'],
                level=0,
                metadata={
                    'source_file': json_file.name,
                    'category': category
                },
                spans=[]
            )
            
            chunks.append(chunk)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(json_files)} files")
                
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    return chunks

def main():
    """Main function to build indices"""
    config_path = "config/config.yaml"
    corpus_dir = Path("data/corpus/final_train")
    
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return
    
    logger.info("Starting index building process...")
    
    # Extract chunks from corpus
    logger.info("Extracting chunks from corpus...")
    chunks = extract_chunks_from_corpus(corpus_dir, max_files=100)  # Adjust max_files as needed
    
    if not chunks:
        logger.error("No chunks extracted from corpus!")
        return
    
    logger.info(f"Extracted {len(chunks)} chunks from corpus")
    
    # Build indices
    logger.info("Building indices...")
    index_builder = HybridIndexBuilder(config_path)
    
    try:
        indices_metadata = index_builder.build_indices(chunks)
        logger.info("Indices built successfully!")
        logger.info(f"Metadata: {indices_metadata}")
        
        # Verify indices
        stats = index_builder.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to build indices: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
