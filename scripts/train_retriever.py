# scripts/train_retriever.py
# BharatWitness retriever training CLI for finetuning dense/sparse models

import argparse
import json
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.seed_utils import set_deterministic_seed
from pipeline.index_build import HybridIndexBuilder


class QADataset(Dataset):
    def __init__(self, qa_pairs: List[Dict[str, Any]], corpus: Dict[str, str]):
        self.qa_pairs = qa_pairs
        self.corpus = corpus
        self.examples = self._create_training_examples()

    def _create_training_examples(self) -> List[InputExample]:
        examples = []

        for qa_pair in self.qa_pairs:
            query = qa_pair['question']
            positive_chunks = qa_pair.get('positive_chunks', [])
            negative_chunks = qa_pair.get('negative_chunks', [])

            for pos_chunk_id in positive_chunks:
                if pos_chunk_id in self.corpus:
                    pos_text = self.corpus[pos_chunk_id]
                    examples.append(InputExample(texts=[query, pos_text], label=1.0))

            for neg_chunk_id in negative_chunks[:5]:
                if neg_chunk_id in self.corpus:
                    neg_text = self.corpus[neg_chunk_id]
                    examples.append(InputExample(texts=[query, neg_text], label=0.0))

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RetrieverTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.retrieval_config = self.config["retrieval"]
        self.model_name = self.retrieval_config["dense_model"]
        self.output_dir = Path(self.config["paths"]["models_root"]) / "retrieval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("bharatwitness.trainer")
        set_deterministic_seed(self.config["system"]["seed"])

    def prepare_training_data(self, qa_file: Path, chunks_file: Path) -> Tuple[List[Dict], Dict[str, str]]:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        corpus = {}
        for chunk_id, chunk_data in chunks_data.items():
            corpus[chunk_id] = chunk_data['text']

        return qa_data, corpus

    def train_dense_retriever(self, qa_data: List[Dict], corpus: Dict[str, str], epochs: int = 3) -> str:
        self.logger.info(f"Training dense retriever with {len(qa_data)} QA pairs")

        model = SentenceTransformer(self.model_name)

        dataset = QADataset(qa_data, corpus)
        train_examples = dataset.examples

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        train_loss = losses.MultipleNegativesRankingLoss(model)

        evaluator = None
        if len(qa_data) > 100:
            eval_queries = {}
            eval_corpus = corpus.copy()
            eval_relevant_docs = {}

            for i, qa_pair in enumerate(qa_data[-20:]):
                query_id = f"q_{i}"
                eval_queries[query_id] = qa_pair['question']
                eval_relevant_docs[query_id] = set(qa_pair.get('positive_chunks', []))

            evaluator = InformationRetrievalEvaluator(
                eval_queries, eval_corpus, eval_relevant_docs,
                name="eval", show_progress_bar=True
            )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=500,
            warmup_steps=100,
            output_path=str(self.output_dir / "dense_finetuned"),
            save_best_model=True
        )

        return str(self.output_dir / "dense_finetuned")

    def create_hard_negatives(self, qa_data: List[Dict], corpus: Dict[str, str], top_k: int = 100) -> List[Dict]:
        self.logger.info("Creating hard negatives using current model")

        index_builder = HybridIndexBuilder()

        chunks = []
        for chunk_id, text in corpus.items():
            from pipeline.chunker import TextChunk
            chunk = TextChunk(
                text=text,
                chunk_id=chunk_id,
                byte_start=0,
                byte_end=len(text.encode()),
                page_nums=[0],
                section_types=['paragraph'],
                level=0,
                metadata={},
                spans=[]
            )
            chunks.append(chunk)

        if not index_builder.load_indices():
            index_builder.build_indices(chunks)

        enhanced_qa_data = []

        for qa_pair in tqdm(qa_data, desc="Adding hard negatives"):
            query = qa_pair['question']
            positive_chunks = set(qa_pair.get('positive_chunks', []))

            if index_builder.dense_encoder and index_builder.faiss_index:
                query_embedding = index_builder.dense_encoder.encode([query])
                scores, indices = index_builder.faiss_index.search(query_embedding.astype('float32'), top_k)

                hard_negatives = []
                chunk_ids = list(corpus.keys())

                for idx in indices[0]:
                    if idx < len(chunk_ids):
                        chunk_id = chunk_ids[idx]
                        if chunk_id not in positive_chunks:
                            hard_negatives.append(chunk_id)

                        if len(hard_negatives) >= 10:
                            break

                qa_pair_enhanced = qa_pair.copy()
                qa_pair_enhanced['negative_chunks'] = hard_negatives
                enhanced_qa_data.append(qa_pair_enhanced)
            else:
                enhanced_qa_data.append(qa_pair)

        return enhanced_qa_data

    def evaluate_retriever(self, qa_data: List[Dict], corpus: Dict[str, str], model_path: Optional[str] = None) -> Dict[
        str, float]:
        if model_path:
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(self.model_name)

        queries = {}
        relevant_docs = {}

        for i, qa_pair in enumerate(qa_data):
            query_id = f"q_{i}"
            queries[query_id] = qa_pair['question']
            relevant_docs[query_id] = set(qa_pair.get('positive_chunks', []))

        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs,
            name="test_eval", show_progress_bar=True
        )

        results = evaluator(model)

        metrics = {
            'ndcg@10': results.get('ndcg@10', 0.0),
            'recall@10': results.get('recall@10', 0.0),
            'recall@100': results.get('recall@100', 0.0),
            'map@10': results.get('map@10', 0.0)
        }

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BharatWitness retriever models")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--qa-data", required=True, help="QA training data JSON file")
    parser.add_argument("--chunks", required=True, help="Chunk corpus JSON file")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--create-negatives", action="store_true", help="Create hard negatives")
    parser.add_argument("--model-path", help="Path to trained model for evaluation")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")

    args = parser.parse_args()

    logger = setup_logging(args.config)

    trainer = RetrieverTrainer(args.config)

    qa_data, corpus = trainer.prepare_training_data(
        Path(args.qa_data),
        Path(args.chunks)
    )

    if args.create_negatives:
        logger.info("Creating hard negatives")
        enhanced_qa_data = trainer.create_hard_negatives(qa_data, corpus)

        output_path = Path(args.qa_data).with_suffix('.enhanced.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_qa_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Enhanced QA data saved to {output_path}")
        return

    if args.eval_only:
        metrics = trainer.evaluate_retriever(qa_data, corpus, args.model_path)
        logger.info(f"Evaluation metrics: {metrics}")
        return

    model_path = trainer.train_dense_retriever(qa_data, corpus, args.epochs)
    logger.info(f"Model trained and saved to {model_path}")

    metrics = trainer.evaluate_retriever(qa_data, corpus, model_path)
    logger.info(f"Final evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
