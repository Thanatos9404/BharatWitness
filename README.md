text
# BharatWitness

Production-grade RAG system for Indian government policy and legal documents.

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```


## Features

- Hybrid dense + sparse retrieval
- OCR-aware document processing
- Temporal reasoning with precedence logic
- Span-level provenance and NLI verification
- Versioned answer diffing
- Comprehensive evaluation suite

## Structure

- `config/` - System configuration
- `ocr/` - OCR and document processing
- `pipeline/` - Core RAG pipeline components
- `evaluation/` - Metrics and ablation studies
- `scripts/` - CLI tools and automation
- `utils/` - Shared utilities

## Requirements

- Python 3.11+
- Windows/Linux/macOS
- CPU-only supported, GPU optional