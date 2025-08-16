# scripts/preprocess_corpus.py
# BharatWitness corpus preprocessing CLI

import argparse
import json
from pathlib import Path
import yaml
from typing import List, Dict, Any
import hashlib
from tqdm import tqdm
import logging

from ocr.ocr_pipeline import OCRPipeline
from utils.logging_utils import setup_logging
from utils.seed_utils import set_deterministic_seed


def calculate_file_hash(file_path: Path) -> str:
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def process_corpus(corpus_root: Path, output_dir: Path, config_path: str) -> None:
    logger = logging.getLogger("bharatwitness.preprocess")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ocr_pipeline = OCRPipeline(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

    files_to_process = []
    for ext in supported_extensions:
        files_to_process.extend(corpus_root.rglob(f"*{ext}"))

    logger.info(f"Found {len(files_to_process)} files to process")

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        for file_path in tqdm(files_to_process, desc="Processing documents"):
            try:
                file_hash = calculate_file_hash(file_path)
                processed_data = ocr_pipeline.process_document(file_path)

                manifest_entry = {
                    'file_path': str(file_path.relative_to(corpus_root)),
                    'file_hash': file_hash,
                    'file_size': file_path.stat().st_size,
                    'processing_timestamp': str(pd.Timestamp.now()),
                    'num_pages': processed_data['metadata'].get('num_pages', 0),
                    'avg_trust_score': sum(page.get('trust_score', 0.0) for page in processed_data['pages']) / len(
                        processed_data['pages']) if processed_data['pages'] else 0.0,
                    'total_sections': sum(len(page.get('sections', [])) for page in processed_data['pages'])
                }

                json.dump(manifest_entry, manifest_file, ensure_ascii=False)
                manifest_file.write('\n')

                processed_file_path = output_dir / f"{file_hash}.json"
                with open(processed_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)

                logger.debug(f"Processed: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

    logger.info(f"Preprocessing complete. Manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess corpus with OCR pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads")

    args = parser.parse_args()

    set_deterministic_seed()
    logger = setup_logging(args.config)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    corpus_root = Path(config["corpus_root"])
    output_dir = Path(args.output)

    if not corpus_root.exists():
        logger.error(f"Corpus root does not exist: {corpus_root}")
        return 1

    logger.info(f"Processing corpus from: {corpus_root}")
    logger.info(f"Output directory: {output_dir}")

    process_corpus(corpus_root, output_dir, args.config)

    return 0


if __name__ == "__main__":
    exit(main())
