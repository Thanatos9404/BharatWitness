# scripts/run_end_to_end.py
# BharatWitness complete pipeline orchestration from ingest to serving

import argparse
import json
import time
from pathlib import Path
import yaml
import logging
from typing import Optional, List, Dict, Any
import subprocess
import sys

from pipeline.index_build import build_indices_from_processed_data
from scripts.preprocess_corpus import process_corpus
from scripts.evaluate import ComprehensiveEvaluator
from utils.logging_utils import setup_logging
from utils.seed_utils import set_deterministic_seed


class EndToEndPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        set_deterministic_seed(self.config["system"]["seed"])
        self.logger = setup_logging(config_path)

        self.corpus_root = Path(self.config["corpus_root"])
        self.processed_root = Path(self.config["paths"]["processed_root"])
        self.models_root = Path(self.config["paths"]["models_root"])
        self.logs_root = Path(self.config["paths"]["logs_root"])

        self._ensure_directories()

    def _ensure_directories(self):
        for directory in [self.processed_root, self.models_root, self.logs_root]:
            directory.mkdir(parents=True, exist_ok=True)

    def run_preprocessing(self, force_reprocess: bool = False) -> bool:
        self.logger.info("Step 1: Document preprocessing and OCR")

        manifest_path = self.processed_root / "manifest.jsonl"

        if manifest_path.exists() and not force_reprocess:
            self.logger.info("Preprocessed data already exists, skipping preprocessing")
            return True

        try:
            process_corpus(
                corpus_root=self.corpus_root,
                output_dir=self.processed_root,
                config_path=self.config_path
            )

            self.logger.info("Preprocessing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return False

    def run_indexing(self, force_rebuild: bool = False) -> bool:
        self.logger.info("Step 2: Index construction")

        index_metadata_path = self.processed_root / "index" / "index_metadata.json"

        if index_metadata_path.exists() and not force_rebuild:
            self.logger.info("Indices already exist, skipping index build")
            return True

        try:
            indices_metadata = build_indices_from_processed_data(
                processed_dir=self.processed_root,
                config_path=self.config_path
            )

            self.logger.info(f"Indexing completed: {indices_metadata}")
            return True

        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            return False

    def run_model_training(self, train_retriever: bool = False, train_nli: bool = False) -> bool:
        self.logger.info("Step 3: Model training (optional)")

        success = True

        if train_retriever:
            self.logger.info("Training retriever models")

            try:
                cmd = [
                    sys.executable, "scripts/train_retriever.py",
                    "--config", self.config_path,
                    "--qa-data", "data/gold_qa/training.json",
                    "--chunks", str(self.processed_root / "chunk_store.json"),
                    "--epochs", "3"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(f"Retriever training failed: {result.stderr}")
                    success = False
                else:
                    self.logger.info("Retriever training completed")

            except Exception as e:
                self.logger.error(f"Retriever training error: {e}")
                success = False

        if train_nli:
            self.logger.info("Training NLI models")

            try:
                cmd = [
                    sys.executable, "scripts/train_nli.py",
                    "--config", self.config_path,
                    "--corpus", str(self.processed_root / "chunk_store.json"),
                    "--synthetic-samples", "1000",
                    "--epochs", "3",
                    "--calibrate"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(f"NLI training failed: {result.stderr}")
                    success = False
                else:
                    self.logger.info("NLI training completed")

            except Exception as e:
                self.logger.error(f"NLI training error: {e}")
                success = False

        return success

    def run_evaluation(self, qa_data_path: Optional[str] = None, max_samples: Optional[int] = None) -> bool:
        self.logger.info("Step 4: System evaluation")

        if not qa_data_path:
            qa_data_path = "data/gold_qa/test.json"

        qa_path = Path(qa_data_path)
        if not qa_path.exists():
            self.logger.warning(f"QA data not found: {qa_path}, creating synthetic data")
            self._create_synthetic_qa_data(qa_path)

        try:
            evaluator = ComprehensiveEvaluator(self.config_path)

            metrics = evaluator.run_comprehensive_evaluation(
                qa_path=qa_path,
                output_prefix="end_to_end",
                max_samples=max_samples,
                run_ablations=False
            )

            targets_met = evaluator.metrics_suite.benchmark_against_targets(metrics)

            self.logger.info("Evaluation Results:")
            self.logger.info(f"  Faithfulness: {metrics.faithfulness:.3f}")
            self.logger.info(f"  Contradiction Rate: {metrics.contradiction_rate:.3f}")
            self.logger.info(f"  nDCG@10: {metrics.ndcg_at_10:.3f}")
            self.logger.info(f"  Span F1: {metrics.span_f1:.3f}")
            self.logger.info(f"  Latency p95: {metrics.latency_p95:.3f}s")

            passed_targets = sum(targets_met.values())
            total_targets = len(targets_met)
            self.logger.info(f"Targets met: {passed_targets}/{total_targets}")

            return all(targets_met.values())

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False

    def _create_synthetic_qa_data(self, output_path: Path):
        synthetic_qa = [
            {
                "question": "What are the current banking regulations for KYC?",
                "answer": "Banks must follow RBI guidelines for customer identification and verification.",
                "as_of_date": "2023-01-01T00:00:00",
                "relevant_spans": [{"start": 0, "end": 100, "document_id": "rbi_kyc_guidelines"}]
            },
            {
                "question": "What is the minimum capital adequacy ratio for banks?",
                "answer": "The minimum capital adequacy ratio is 9% as per Basel III norms.",
                "as_of_date": "2023-01-01T00:00:00",
                "relevant_spans": [{"start": 200, "end": 300, "document_id": "basel_capital_norms"}]
            },
            {
                "question": "What are the digital payment regulations?",
                "answer": "Digital payments are regulated under the Payment and Settlement Systems Act.",
                "as_of_date": "2023-01-01T00:00:00",
                "relevant_spans": [{"start": 400, "end": 500, "document_id": "digital_payment_act"}]
            }
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(synthetic_qa, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Created synthetic QA data at {output_path}")

    def start_server(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> bool:
        self.logger.info(f"Step 5: Starting API server on {host}:{port}")

        try:
            cmd = [
                sys.executable, "main.py",
                "--host", host,
                "--port", str(port),
                "--config", self.config_path
            ]

            if reload:
                cmd.append("--reload")

            self.logger.info(f"Server command: {' '.join(cmd)}")

            process = subprocess.Popen(cmd)

            self.logger.info(f"Server started with PID {process.pid}")
            self.logger.info(f"API documentation available at http://{host}:{port}/docs")

            try:
                process.wait()
            except KeyboardInterrupt:
                self.logger.info("Shutting down server")
                process.terminate()
                process.wait()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    def run_complete_pipeline(
            self,
            force_reprocess: bool = False,
            force_rebuild: bool = False,
            train_models: bool = False,
            run_eval: bool = True,
            start_server: bool = True,
            qa_data_path: Optional[str] = None,
            server_host: str = "127.0.0.1",
            server_port: int = 8000
    ) -> bool:

        self.logger.info("Starting complete BharatWitness pipeline")
        start_time = time.time()

        steps_success = []

        steps_success.append(self.run_preprocessing(force_reprocess))

        if steps_success[-1]:
            steps_success.append(self.run_indexing(force_rebuild))
        else:
            self.logger.error("Preprocessing failed, stopping pipeline")
            return False

        if train_models and steps_success[-1]:
            steps_success.append(self.run_model_training(train_retriever=True, train_nli=True))

        if run_eval and all(steps_success):
            steps_success.append(self.run_evaluation(qa_data_path))

        total_time = time.time() - start_time
        successful_steps = sum(steps_success)
        total_steps = len(steps_success)

        self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        self.logger.info(f"Successful steps: {successful_steps}/{total_steps}")

        if all(steps_success) and start_server:
            self.logger.info("All steps successful, starting server")
            return self.start_server(server_host, server_port)

        return all(steps_success)


def main():
    parser = argparse.ArgumentParser(description="BharatWitness end-to-end pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of documents")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding of indices")
    parser.add_argument("--train-models", action="store_true", help="Train retriever and NLI models")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--no-server", action="store_true", help="Don't start API server")
    parser.add_argument("--qa-data", help="Path to QA evaluation data")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")

    args = parser.parse_args()

    pipeline = EndToEndPipeline(args.config)

    success = pipeline.run_complete_pipeline(
        force_reprocess=args.force_reprocess,
        force_rebuild=args.force_rebuild,
        train_models=args.train_models,
        run_eval=not args.skip_eval,
        start_server=not args.no_server,
        qa_data_path=args.qa_data,
        server_host=args.host,
        server_port=args.port
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
