# utils/logging_utils.py
# BharatWitness logging configuration

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml


def setup_logging(
        config_path: str = "config/config.yaml",
        logger_name: str = "bharatwitness"
) -> logging.Logger:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    log_level = getattr(logging, config["system"]["log_level"].upper())
    logs_root = Path(config["paths"]["logs_root"])
    logs_root.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        logs_root / f"bharatwitness_{timestamp}.log"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
