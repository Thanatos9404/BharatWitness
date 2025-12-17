#!/usr/bin/env python3
"""
Test retrieval to debug why no results are returned
"""

import logging
from pipeline.retrieval import HybridRetriever, QueryContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_retrieval():
    """Test the retrieval system"""
    logger.info("Initializing retriever...")
    retriever = HybridRetriever()
    
    logger.info(f"Chunk store size: {len(retriever.chunk_store)}")
    logger.info(f"TF-IDF vectorizer: {retriever.tfidf_vectorizer is not None}")
    logger.info(f"Dense embeddings: {retriever.dense_embeddings is not None if hasattr(retriever, 'dense_embeddings') else 'N/A'}")
    logger.info(f"BM25 index: {retriever.bm25_index is not None}")
    
    if retriever.chunk_store:
        logger.info(f"Sample chunk IDs: {list(retriever.chunk_store.keys())[:3]}")
    
    # Test query
    query = "What is science?"
    logger.info(f"\nTesting query: '{query}'")
    
    # Test individual search methods
    logger.info("\n=== Testing individual search methods ===")
    dense_results = retriever._dense_search(query, 10)
    logger.info(f"Dense search results: {len(dense_results)}")
    if dense_results:
        logger.info(f"Top dense: {dense_results[:2]}")
    
    bm25_results = retriever._bm25_search(query, 10)
    logger.info(f"BM25 search results: {len(bm25_results)}")
    if bm25_results:
        logger.info(f"Top BM25: {bm25_results[:2]}")
    
    logger.info("\n=== Testing full retrieval ===")
    query_context = QueryContext(
        query=query,
        as_of_date=None,
        language_filter=["en"],
        section_type_filter=None,
        confidence_threshold=0.1,  # Lower threshold for testing
        max_results=10
    )
    
    results = retriever.retrieve(query_context)
    
    logger.info(f"Retrieved {len(results)} results")
    
    for idx, result in enumerate(results[:3]):
        logger.info(f"\nResult {idx + 1}:")
        logger.info(f"  Chunk ID: {result.chunk_id}")
        logger.info(f"  Score: {result.score:.4f}")
        logger.info(f"  Confidence: {result.confidence:.4f}")
        logger.info(f"  Text preview: {result.text[:150]}...")

if __name__ == "__main__":
    test_retrieval()
