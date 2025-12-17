import logging
# pipeline/retrieval.py
# BharatWitness hybrid retrieval with dense, sparse, and BM25 fusion

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import yaml

from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import dateparser
from collections import defaultdict
import heapq
import regex as re

from pipeline.index_build import HybridIndexBuilder
from utils.span_utils import TextSpan, SpanManager
from utils.temporal_utils import TemporalFilter


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    dense_score: float
    sparse_score: float
    bm25_score: float
    metadata: Dict[str, Any]
    spans: List[Dict[str, Any]]
    confidence: float


@dataclass
class QueryContext:
    query: str
    as_of_date: Optional[datetime]
    language_filter: Optional[List[str]]
    section_type_filter: Optional[List[str]]
    confidence_threshold: float
    max_results: int


class HybridRetriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.retrieval_config = self.config["retrieval"]
        self.paths_config = self.config["paths"]

        self.top_k = self.retrieval_config["top_k"]
        self.hybrid_alpha = self.retrieval_config["hybrid_alpha"]

        self.index_root = Path(self.paths_config["index_root"])
        self.logger = logging.getLogger("bharatwitness.retrieval")

        self.index_builder = HybridIndexBuilder(config_path)
        self.span_manager = SpanManager()
        self.temporal_filter = TemporalFilter()

        self.dense_encoder = None
        self.faiss_index = None
        self.sparse_vectors = None
        self.sparse_vocab = None
        self.bm25_index = None
        self.chunk_store = {}
        self.tfidf_vectorizer = None  # Fallback for dense search
        self.dense_embeddings = None  # TF-IDF embeddings

        self.reranker = None
        self._load_indices()

    def _load_indices(self) -> bool:
        try:
            if not self.index_builder.load_indices():
                self.logger.warning("Could not load pre-built indices")
                return False

            self.dense_encoder = self.index_builder.dense_encoder
            self.faiss_index = self.index_builder.faiss_index
            self.bm25_index = self.index_builder.bm25_index
            self.chunk_store = self.index_builder.chunk_store

            # Load TF-IDF fallback if available
            tfidf_vectorizer_path = self.index_root / "tfidf_vectorizer.pkl"
            if tfidf_vectorizer_path.exists():
                with open(tfidf_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.logger.info("Loaded TF-IDF vectorizer")
                
            # Load dense embeddings (TF-IDF or sentence embeddings)
            embeddings_path = self.index_root / "dense_embeddings.npy"
            if embeddings_path.exists():
                self.dense_embeddings = np.load(embeddings_path)
                self.logger.info(f"Loaded dense embeddings: {self.dense_embeddings.shape}")

            sparse_path = self.index_root / "sparse_vectors.npy"
            if sparse_path.exists():
                self.sparse_vectors = np.load(sparse_path)

                vocab_path = self.index_root / "sparse_vocab.json"
                if vocab_path.exists():
                    with open(vocab_path, 'r') as f:
                        self.sparse_vocab = json.load(f)

            self.logger.info("Successfully loaded all retrieval indices")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load indices: {e}")
            return False

    def retrieve(self, query_context: QueryContext) -> List[RetrievalResult]:
        query = query_context.query

        dense_results = []
        sparse_results = []
        bm25_results = []

        if (self.faiss_index and self.dense_encoder) or (self.tfidf_vectorizer and self.dense_embeddings is not None):
            dense_results = self._dense_search(query, query_context.max_results)

        if self.sparse_vectors is not None:
            sparse_results = self._sparse_search(query, query_context.max_results)

        if self.bm25_index:
            bm25_results = self._bm25_search(query, query_context.max_results)

        fused_results = self._fuse_results(dense_results, sparse_results, bm25_results)

        filtered_results = self._apply_filters(fused_results, query_context)

        if len(filtered_results) > query_context.max_results:
            filtered_results = filtered_results[:query_context.max_results]

        if self.reranker and len(filtered_results) > 10:
            filtered_results = self._rerank_results(query, filtered_results)

        return filtered_results

    def _dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        # Try TF-IDF fallback first if FAISS not available
        if self.tfidf_vectorizer and self.dense_embeddings is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([query]).toarray()[0]
                similarities = np.dot(self.dense_embeddings, query_vector)
                top_indices = np.argsort(similarities)[-k:][::-1]
                
                results = []
                chunk_ids = list(self.chunk_store.keys())
                
                for idx in top_indices:
                    if idx < len(chunk_ids) and similarities[idx] > 0:
                        chunk_id = chunk_ids[idx]
                        results.append((chunk_id, float(similarities[idx])))
                
                return results
            except Exception as e:
                self.logger.error(f"TF-IDF dense search failed: {e}")
        
        # Fall back to FAISS if available
        if not self.dense_encoder or not self.faiss_index:
            return []

        try:
            query_embedding = self.dense_encoder.encode([query], normalize_embeddings=True)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)

            results = []
            chunk_ids = list(self.chunk_store.keys())

            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunk_ids) and score > 0:
                    chunk_id = chunk_ids[idx]
                    results.append((chunk_id, float(score)))

            return results

        except Exception as e:
            self.logger.error(f"Dense search failed: {e}")
            return []

    def _sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.sparse_vectors is None:
            return []

        try:
            query_vector = self._encode_sparse_query(query)
            if query_vector is None:
                return []

            similarities = np.dot(self.sparse_vectors, query_vector)
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            chunk_ids = list(self.chunk_store.keys())

            for idx in top_indices:
                if idx < len(chunk_ids) and similarities[idx] > 0:
                    chunk_id = chunk_ids[idx]
                    results.append((chunk_id, float(similarities[idx])))

            return results

        except Exception as e:
            self.logger.error(f"Sparse search failed: {e}")
            return []

    def _bm25_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self.bm25_index:
            return []

        try:
            import nltk
            from nltk.tokenize import word_tokenize

            query_tokens = word_tokenize(query.lower())
            scores = self.bm25_index.get_scores(query_tokens)

            top_indices = np.argsort(scores)[-k:][::-1]

            results = []
            chunk_ids = list(self.chunk_store.keys())

            for idx in top_indices:
                if idx < len(chunk_ids) and scores[idx] > 0:
                    chunk_id = chunk_ids[idx]
                    results.append((chunk_id, float(scores[idx])))

            return results

        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []

    def _encode_sparse_query(self, query: str) -> Optional[np.ndarray]:
        if not self.sparse_vocab:
            return None

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch.nn.functional as F

            tokenizer = AutoTokenizer.from_pretrained(self.retrieval_config["sparse_model"])
            model = AutoModelForMaskedLM.from_pretrained(self.retrieval_config["sparse_model"])

            if torch.cuda.is_available():
                model = model.cuda()

            model.eval()

            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                sparse_vector = \
                torch.max(torch.log(1 + torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1), dim=1)[0]
                return sparse_vector.cpu().numpy().flatten()

        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer_path = self.index_root / "tfidf_vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                    query_vector = vectorizer.transform([query]).toarray()
                    return query_vector.flatten()

            return None
        except Exception as e:
            self.logger.error(f"Sparse query encoding failed: {e}")
            return None

    def _fuse_results(self, dense_results: List[Tuple[str, float]],
                      sparse_results: List[Tuple[str, float]],
                      bm25_results: List[Tuple[str, float]]) -> List[RetrievalResult]:

        score_maps = {
            'dense': dict(dense_results),
            'sparse': dict(sparse_results),
            'bm25': dict(bm25_results)
        }

        all_chunk_ids = set()
        for result_list in [dense_results, sparse_results, bm25_results]:
            all_chunk_ids.update([chunk_id for chunk_id, _ in result_list])

        fused_results = []

        for chunk_id in all_chunk_ids:
            if chunk_id not in self.chunk_store:
                continue

            chunk_data = self.chunk_store[chunk_id]

            dense_score = score_maps['dense'].get(chunk_id, 0.0)
            sparse_score = score_maps['sparse'].get(chunk_id, 0.0)
            bm25_score = score_maps['bm25'].get(chunk_id, 0.0)

            dense_score_norm = self._normalize_score(dense_score, [s for _, s in dense_results])
            sparse_score_norm = self._normalize_score(sparse_score, [s for _, s in sparse_results])
            bm25_score_norm = self._normalize_score(bm25_score, [s for _, s in bm25_results])

            hybrid_score = self._compute_hybrid_score(dense_score_norm, sparse_score_norm, bm25_score_norm)

            confidence = self._compute_confidence(dense_score_norm, sparse_score_norm, bm25_score_norm)

            result = RetrievalResult(
                chunk_id=chunk_id,
                text=chunk_data['text'],
                score=hybrid_score,
                dense_score=dense_score_norm,
                sparse_score=sparse_score_norm,
                bm25_score=bm25_score_norm,
                metadata=chunk_data.get('metadata', {}),
                spans=chunk_data.get('spans', []),
                confidence=confidence
            )

            fused_results.append(result)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _normalize_score(self, score: float, all_scores: List[float]) -> float:
        if not all_scores or max(all_scores) == 0:
            return 0.0

        min_score = min(all_scores)
        max_score = max(all_scores)

        if max_score == min_score:
            return 1.0 if score > 0 else 0.0

        return (score - min_score) / (max_score - min_score)

    def _compute_hybrid_score(self, dense_score: float, sparse_score: float, bm25_score: float) -> float:
        alpha = self.hybrid_alpha
        beta = 0.2

        learned_score = alpha * dense_score + (1 - alpha) * sparse_score
        final_score = (1 - beta) * learned_score + beta * bm25_score

        return final_score

    def _compute_confidence(self, dense_score: float, sparse_score: float, bm25_score: float) -> float:
        agreement_score = 1.0 - abs(dense_score - sparse_score)
        avg_score = (dense_score + sparse_score + bm25_score) / 3

        confidence = 0.7 * avg_score + 0.3 * agreement_score
        return min(1.0, max(0.0, confidence))

    def _apply_filters(self, results: List[RetrievalResult], query_context: QueryContext) -> List[RetrievalResult]:
        filtered_results = []

        for result in results:
            if result.confidence < query_context.confidence_threshold:
                continue

            if query_context.language_filter:
                chunk_languages = result.metadata.get('languages', ['en'])  # Default to 'en' if not specified
                if not any(lang in query_context.language_filter for lang in chunk_languages):
                    continue

            if query_context.section_type_filter:
                chunk_sections = result.metadata.get('section_types', [])
                if not any(section in query_context.section_type_filter for section in chunk_sections):
                    continue

            if query_context.as_of_date:
                if not self.temporal_filter.is_valid_as_of(result, query_context.as_of_date):
                    continue

            filtered_results.append(result)

        return filtered_results

    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not self.reranker:
            return results

        try:
            texts = [result.text for result in results]
            query_text_pairs = [(query, text) for text in texts]

            rerank_scores = self.reranker.predict(query_text_pairs)

            for i, result in enumerate(results):
                if i < len(rerank_scores):
                    result.score = 0.7 * result.score + 0.3 * rerank_scores[i]

            results.sort(key=lambda x: x.score, reverse=True)

        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")

        return results

    def search_with_rrf(self, query_context: QueryContext) -> List[RetrievalResult]:
        query = query_context.query
        k = min(query_context.max_results * 2, 100)

        dense_results = self._dense_search(query, k)
        sparse_results = self._sparse_search(query, k)
        bm25_results = self._bm25_search(query, k)

        rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results, bm25_results)

        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id not in self.chunk_store:
                continue

            chunk_data = self.chunk_store[chunk_id]

            dense_score = dict(dense_results).get(chunk_id, 0.0)
            sparse_score = dict(sparse_results).get(chunk_id, 0.0)
            bm25_score = dict(bm25_results).get(chunk_id, 0.0)

            confidence = self._compute_confidence(
                self._normalize_score(dense_score, [s for _, s in dense_results]),
                self._normalize_score(sparse_score, [s for _, s in sparse_results]),
                self._normalize_score(bm25_score, [s for _, s in bm25_results])
            )

            result = RetrievalResult(
                chunk_id=chunk_id,
                text=chunk_data['text'],
                score=rrf_score,
                dense_score=dense_score,
                sparse_score=sparse_score,
                bm25_score=bm25_score,
                metadata=chunk_data.get('metadata', {}),
                spans=chunk_data.get('spans', []),
                confidence=confidence
            )

            fused_results.append(result)

        fused_results.sort(key=lambda x: x.score, reverse=True)

        filtered_results = self._apply_filters(fused_results, query_context)

        return filtered_results[:query_context.max_results]

    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[str, float]],
                                sparse_results: List[Tuple[str, float]],
                                bm25_results: List[Tuple[str, float]],
                                k: int = 60) -> Dict[str, float]:

        rrf_scores = defaultdict(float)

        for rank, (chunk_id, _) in enumerate(dense_results):
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        for rank, (chunk_id, _) in enumerate(sparse_results):
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        return dict(rrf_scores)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        stats = self.index_builder.get_index_stats()

        stats.update({
            'hybrid_alpha': self.hybrid_alpha,
            'top_k': self.top_k,
            'reranker_available': self.reranker is not None,
            'temporal_filtering': True
        })

        return stats

    def create_text_spans(self, results: List[RetrievalResult], query: str) -> List[TextSpan]:
        spans = []

        for result in results:
            for span_data in result.spans:
                span = self.span_manager.create_span(
                    text=span_data['text'],
                    start=span_data['byte_start'],
                    end=span_data['byte_end'],
                    document_id=result.chunk_id.split('_')[0],
                    page_num=span_data['page_num'],
                    section_id=span_data.get('section_type', 'unknown'),
                    confidence=result.confidence,
                    metadata={
                        'retrieval_score': result.score,
                        'query': query,
                        'chunk_id': result.chunk_id
                    }
                )
                spans.append(span)

        return spans
