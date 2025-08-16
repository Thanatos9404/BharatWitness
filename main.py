# main.py
# BharatWitness main entry point

import argparse
from pathlib import Path
from utils.seed_utils import set_deterministic_seed
from utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="BharatWitness RAG System")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    seed = set_deterministic_seed(args.seed)
    logger = setup_logging(args.config)

    logger.info(f"BharatWitness initialized with seed {seed}")

    if args.offline:
        logger.info("Running in offline mode")

    logger.info("System ready")


if __name__ == "__main__":
    main()
