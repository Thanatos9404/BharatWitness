#!/usr/bin/env python3
"""
Quick lightweight script to build indices with smaller models for testing
"""

import json
import logging
from pathlib import Path
import hashlib
import pickle
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_simple_indices():
    """Build basic BM25 and simple vector indices"""
    
    corpus_dir = Path("data/corpus/final_train")
    index_root = Path("indices")
    index_root.mkdir(parents=True, exist_ok=True)
    
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return False
    
    logger.info("Loading corpus data...")
    
    # Load corpus documents
    json_files = list(corpus_dir.glob("*.json"))[:100]  # Use first 100 files for quick testing
    logger.info(f"Found {len(json_files)} JSON files")
    
    documents = []
    chunk_store = {}
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            category = data.get('category', 'Unknown Category')
            search_results = data.get('search_results', [])
            
            # Create document text
            doc_text = f"Category: {category}\n\n"
            
            for result in search_results[:3]:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                if title or snippet:
                    doc_text += f"{title}\n{snippet}\n\n"
            
            chunk_id = f"chunk_{idx}_{hashlib.md5(json_file.name.encode()).hexdigest()[:8]}"
            
            documents.append(doc_text.strip())
            
            chunk_store[chunk_id] = {
                'text': doc_text.strip(),
                'byte_start': 0,
                'byte_end': len(doc_text.encode('utf-8')),
                'page_nums': [0],
                'section_types': ['general'],
                'level': 0,
                'metadata': {
                    'source_file': json_file.name,
                    'category': category
                },
                'spans': []
            }
            
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(json_files)} files")
                
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Build BM25 index
    logger.info("Building BM25 index...")
    try:
        from rank_bm25 import BM25Okapi
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        from nltk.tokenize import word_tokenize
        
        tokenized_docs = []
        for doc in documents:
            try:
                tokens = word_tokenize(doc.lower())
                tokenized_docs.append(tokens)
            except:
                tokenized_docs.append(doc.lower().split())
        
        bm25_index = BM25Okapi(tokenized_docs)
        
        bm25_path = index_root / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        
        logger.info("BM25 index built successfully")
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return False
    
    # Build simple TF-IDF index as fallback for dense embeddings
    logger.info("Building TF-IDF index...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Save as dense embeddings
        embeddings = tfidf_matrix.toarray().astype('float32')
        embeddings_path = index_root / "dense_embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Also save vectorizer
        vectorizer_path = index_root / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        logger.info("TF-IDF index built successfully")
    except Exception as e:
        logger.error(f"Failed to build TF-IDF index: {e}")
        return False
    
    # Save chunk store
    logger.info("Saving chunk store...")
    chunks_path = index_root / "chunk_store.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_store, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    logger.info("Saving metadata...")
    metadata = {
        "dense": {
            "index_type": "tfidf",
            "model_name": "tfidf_vectorizer",
            "num_chunks": len(documents),
            "embedding_dim": embeddings.shape[1],
            "created_timestamp": datetime.now().isoformat(),
            "chunk_ids": list(chunk_store.keys())
        },
        "bm25": {
            "index_type": "bm25",
            "model_name": "bm25_okapi",
            "num_chunks": len(documents),
            "created_timestamp": datetime.now().isoformat(),
            "chunk_ids": list(chunk_store.keys())
        }
    }
    
    metadata_path = index_root / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info(f"Total documents indexed: {len(documents)}")
    logger.info(f"Index directory: {index_root.absolute()}")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = build_simple_indices()
    if success:
        logger.info("✓ Indices built successfully!")
    else:
        logger.error("✗ Failed to build indices")
