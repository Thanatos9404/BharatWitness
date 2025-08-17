# pipeline/index_build.py
# BharatWitness index construction with FAISS dense, SPLADE sparse, and BM25 fallback

import pandas as pd
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import yaml
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import hashlib
from dataclasses import dataclass
from bharatwitness.pipeline.chunker import TextChunk
from bharatwitness.pipeline.segment import DocumentSegmenter
from bharatwitness.pipeline.chunker import DocumentChunker
from bharatwitness.utils.seed_utils import set_deterministic_seed


@dataclass
class IndexMetadata:
    index_type: str
    model_name: str
    num_chunks: int
    embedding_dim: int
    created_timestamp: str
    config_hash: str
    chunk_ids: List[str]


class HybridIndexBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.retrieval_config = self.config["retrieval"]
        self.paths_config = self.config["paths"]

        self.dense_model_name = self.retrieval_config["dense_model"]
        self.sparse_model_name = self.retrieval_config["sparse_model"]
        self.top_k = self.retrieval_config["top_k"]

        self.index_root = Path(self.paths_config["index_root"])
        self.index_root.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("bharatwitness.index")

        set_deterministic_seed(self.config["system"]["seed"])

        self.dense_encoder = None
        self.bm25_index = None
        self.faiss_index = None
        self.chunk_store = {}

    def build_indices(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        self.logger.info(f"Building indices for {len(chunks)} chunks")

        chunk_texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        self._store_chunks(chunks)

        indices_metadata = {}

        try:
            dense_metadata = self._build_dense_index(chunk_texts, chunk_ids)
            indices_metadata["dense"] = dense_metadata
        except Exception as e:
            self.logger.warning(f"Dense index build failed: {e}")

        try:
            sparse_metadata = self._build_sparse_index(chunk_texts, chunk_ids)
            indices_metadata["sparse"] = sparse_metadata
        except Exception as e:
            self.logger.warning(f"Sparse index build failed: {e}")

        try:
            bm25_metadata = self._build_bm25_index(chunk_texts, chunk_ids)
            indices_metadata["bm25"] = bm25_metadata
        except Exception as e:
            self.logger.warning(f"BM25 index build failed: {e}")

        self._save_index_metadata(indices_metadata)

        return indices_metadata

    def _build_dense_index(self, texts: List[str], chunk_ids: List[str]) -> IndexMetadata:
        self.logger.info(f"Building dense index with model: {self.dense_model_name}")

        if self.dense_encoder is None:
            self.dense_encoder = SentenceTransformer(self.dense_model_name)
            if torch.cuda.is_available():
                self.dense_encoder = self.dense_encoder.cuda()

        embeddings = self.dense_encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embedding_dim = embeddings.shape[1]

        self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.faiss_index.hnsw.efConstruction = 40
        self.faiss_index.hnsw.efSearch = 16

        self.faiss_index.add(embeddings.astype('float32'))

        dense_index_path = self.index_root / "dense_index.faiss"
        faiss.write_index(self.faiss_index, str(dense_index_path))

        embeddings_path = self.index_root / "dense_embeddings.npy"
        np.save(embeddings_path, embeddings)

        return IndexMetadata(
            index_type="dense_hnsw",
            model_name=self.dense_model_name,
            num_chunks=len(texts),
            embedding_dim=embedding_dim,
            created_timestamp=str(pd.Timestamp.now()),
            config_hash=self._get_config_hash(),
            chunk_ids=chunk_ids
        )

    def _build_sparse_index(self, texts: List[str], chunk_ids: List[str]) -> IndexMetadata:
        self.logger.info("Building SPLADE sparse index")

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch.nn.functional as F

            tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
            model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_name)

            if torch.cuda.is_available():
                model = model.cuda()

            model.eval()
            sparse_vectors = []

            for text in tqdm(texts, desc="Encoding SPLADE"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    sparse_vector = \
                    torch.max(torch.log(1 + torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1), dim=1)[0]
                    sparse_vectors.append(sparse_vector.cpu().numpy())

            sparse_matrix = np.vstack(sparse_vectors)

            sparse_index_path = self.index_root / "sparse_vectors.npy"
            np.save(sparse_index_path, sparse_matrix)

            vocab_path = self.index_root / "sparse_vocab.json"
            with open(vocab_path, 'w') as f:
                json.dump(tokenizer.get_vocab(), f)

        except ImportError:
            self.logger.warning("SPLADE dependencies not available, using TF-IDF fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
            sparse_matrix = vectorizer.fit_transform(texts).toarray()

            sparse_index_path = self.index_root / "sparse_vectors.npy"
            np.save(sparse_index_path, sparse_matrix)

            vectorizer_path = self.index_root / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

        return IndexMetadata(
            index_type="sparse_learned",
            model_name=self.sparse_model_name,
            num_chunks=len(texts),
            embedding_dim=sparse_matrix.shape[1],
            created_timestamp=str(pd.Timestamp.now()),
            config_hash=self._get_config_hash(),
            chunk_ids=chunk_ids
        )

    def _build_bm25_index(self, texts: List[str], chunk_ids: List[str]) -> IndexMetadata:
        self.logger.info("Building BM25 index")

        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        from nltk.tokenize import word_tokenize

        tokenized_texts = []
        for text in texts:
            try:
                tokens = word_tokenize(text.lower())
                tokenized_texts.append(tokens)
            except:
                tokenized_texts.append(text.lower().split())

        self.bm25_index = BM25Okapi(tokenized_texts)

        bm25_path = self.index_root / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)

        return IndexMetadata(
            index_type="bm25",
            model_name="bm25_okapi",
            num_chunks=len(texts),
            embedding_dim=0,
            created_timestamp=str(pd.Timestamp.now()),
            config_hash=self._get_config_hash(),
            chunk_ids=chunk_ids
        )

    def _store_chunks(self, chunks: List[TextChunk]):
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = {
                'text': chunk.text,
                'byte_start': chunk.byte_start,
                'byte_end': chunk.byte_end,
                'page_nums': chunk.page_nums,
                'section_types': chunk.section_types,
                'level': chunk.level,
                'metadata': chunk.metadata,
                'spans': chunk.spans
            }

        chunks_path = self.index_root / "chunk_store.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_store, f, ensure_ascii=False, indent=2)

    def _get_config_hash(self) -> str:
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _save_index_metadata(self, metadata: Dict[str, Any]):
        metadata_path = self.index_root / "index_metadata.json"
        serializable_metadata = {}

        for index_type, meta in metadata.items():
            if hasattr(meta, '__dict__'):
                serializable_metadata[index_type] = meta.__dict__
            else:
                serializable_metadata[index_type] = meta

        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, default=str)

    def load_indices(self) -> bool:
        try:
            metadata_path = self.index_root / "index_metadata.json"
            if not metadata_path.exists():
                return False

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if "dense" in metadata:
                dense_index_path = self.index_root / "dense_index.faiss"
                if dense_index_path.exists():
                    self.faiss_index = faiss.read_index(str(dense_index_path))
                    if self.dense_encoder is None:
                        self.dense_encoder = SentenceTransformer(self.dense_model_name)

            if "bm25" in metadata:
                bm25_path = self.index_root / "bm25_index.pkl"
                if bm25_path.exists():
                    with open(bm25_path, 'rb') as f:
                        self.bm25_index = pickle.load(f)

            chunks_path = self.index_root / "chunk_store.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunk_store = json.load(f)

            self.logger.info("Successfully loaded existing indices")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load indices: {e}")
            return False

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.chunk_store.get(chunk_id)

    def get_index_stats(self) -> Dict[str, Any]:
        stats = {
            'total_chunks': len(self.chunk_store),
            'indices_available': []
        }

        if self.faiss_index is not None:
            stats['indices_available'].append('dense')
            stats['dense_index_size'] = self.faiss_index.ntotal

        if self.bm25_index is not None:
            stats['indices_available'].append('bm25')

        sparse_path = self.index_root / "sparse_vectors.npy"
        if sparse_path.exists():
            stats['indices_available'].append('sparse')

        return stats


def build_indices_from_processed_data(processed_dir: Path, config_path: str) -> Dict[str, Any]:
    segmenter = DocumentSegmenter(config_path)
    chunker = DocumentChunker(config_path)
    index_builder = HybridIndexBuilder(config_path)

    all_chunks = []
    manifest_path = processed_dir / "manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    logger = logging.getLogger("bharatwitness.build")
    logger.info("Loading processed documents for indexing")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_entry = json.loads(line.strip())
            file_hash = manifest_entry['file_hash']

            processed_file_path = processed_dir / f"{file_hash}.json"
            if not processed_file_path.exists():
                logger.warning(f"Processed file not found: {processed_file_path}")
                continue

            with open(processed_file_path, 'r', encoding='utf-8') as doc_file:
                document_data = json.load(doc_file)

            sections = segmenter.segment_document(document_data)
            chunks = chunker.chunk_document(sections, file_hash)
            all_chunks.extend(chunks)

    logger.info(f"Total chunks to index: {len(all_chunks)}")

    indices_metadata = index_builder.build_indices(all_chunks)
    return indices_metadata
