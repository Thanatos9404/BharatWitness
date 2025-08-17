# BharatWitness

Production-grade RAG system for Indian government policy and legal documents with temporal reasoning, span-level provenance, and hybrid retrieval.

## Features

### Core Capabilities
- **Hybrid Dense + Sparse Semantic Retrieval** optimized for Indian multi-script documents including OCR-corrected scanned PDFs
- **Temporal Reasoning and Legal Precedence Logic** - answers respect effective dates, repeals, supersessions, and cross-document conflict resolution
- **Span-level Provenance and NLI-based Claim Verification** to minimize hallucinations with attributable source citations
- **Versioned Answer Diffing** to visualize how laws or policies evolve over time
- **Robust OCR Trust Gating** for noisy, multi-language inputs with fallback mechanisms and Unicode normalization

### Technical Architecture
- **Multi-modal Document Processing**: PaddleOCR with layout preservation for scanned documents
- **Hierarchical Text Segmentation**: Layout-aware chunking preserving document structure
- **Hybrid Retrieval Stack**: SPLADE sparse + dense embeddings (multilingual-e5) + BM25 fallback with HNSW-FAISS indexing
- **Temporal Engine**: Date-aware filtering with precedence resolution for legal documents
- **Comprehensive Evaluation Suite**: Faithfulness, robustness, latency, and calibration metrics

## Quick Start

### Prerequisites
- Python 3.11+
- Windows/Linux/macOS
- CPU-only supported, GPU optional for acceleration

### Installation
```
git clone <repository-url>
cd bharatwitness
pip install -r requirements.txt
```

### Basic Usage
Process raw documents
python scripts/preprocess_corpus.py --config config/config.yaml

Build search indices
python -c "from pipeline.index_build import build_indices_from_processed_data; build_indices_from_processed_data('data/processed', 'config/config.yaml')"

Start system
python main.py --config config/config.yaml

text

## Project Structure
```
bharatwitness/
├── config/ # System configuration
│ └── config.yaml # Main configuration file
├── data/ # Data storage
│ ├── gold_qa/ # Ground truth QA pairs
│ └── processed/ # Processed documents and indices
├── docs/ # Documentation
├── evaluation/ # Metrics and evaluation scripts
├── logs/ # System logs and audit trails
├── models/ # Trained model artifacts
│ ├── nli/ # NLI verification models
│ └── retrieval/ # Retrieval models
├── ocr/ # OCR processing pipeline
│ └── ocr_pipeline.py # Multi-lingual OCR with layout analysis
├── pipeline/ # Core RAG pipeline
│ ├── segment.py # Document segmentation
│ ├── chunker.py # Text chunking with hierarchy
│ ├── index_build.py # Hybrid index construction
│ └── retrieval.py # Multi-modal retrieval engine
├── scripts/ # CLI automation tools
│ ├── preprocess_corpus.py # Document preprocessing
│ └── train_retriever.py # Model training
├── tests/ # Test suite
├── utils/ # Shared utilities
│ ├── layout.py # Layout analysis
│ ├── ocr_utils.py # OCR processing helpers
│ ├── span_utils.py # Span management and provenance
│ └── temporal_utils.py # Temporal reasoning
└── main.py # Main entry point
```

## Configuration

The system is configured via `config/config.yaml`:

- **corpus_root**: Path to raw document folder 
- **retrieval**: Dense/sparse model settings, hybrid weights
- **ocr**: Trust thresholds, language support
- **temporal**: Precedence rules, date filtering
- **evaluation**: Metrics and thresholds

## Advanced Features

### Temporal Reasoning
```
from pipeline.retrieval import QueryContext
from datetime import datetime

query_context = QueryContext(
query="Banking regulations for KYC",
as_of_date=datetime(2023, 6, 1),
confidence_threshold=0.8,
max_results=10
)
```


### Multi-language Support
- Hindi-English code-mixed queries
- Devanagari and Latin script processing
- Unicode normalization for Indic languages

### Evaluation Metrics
- **Faithfulness**: Attributable claims ≥95%
- **Contradiction Rate**: <2%
- **Retrieval Quality**: nDCG@10 ≥0.85
- **Temporal Accuracy**: ≥98% date-valid spans
- **Latency**: p95 ≤3.0 seconds end-to-end

## Training Custom Models

### Dense Retriever Training
```
python scripts/train_retriever.py
--qa-data data/gold_qa/training.json
--chunks data/processed/chunk_store.json
--epochs 3
```


### Hard Negative Mining
```
python scripts/train_retriever.py
--qa-data data/gold_qa/training.json
--chunks data/processed/chunk_store.json
--create-negatives
```

## Development

### Running Tests
```
pytest tests/ -v
```

### Code Quality
- Type hints enforced throughout
- No inline comments in production code
- Deterministic seeding for reproducibility
- Comprehensive logging with audit trails

## License

Production-grade research system for hackathon and academic use.

## Contributing

>This system prioritizes reproducibility, operational readiness, and measurable advances over generic RAG implementations. All contributions should include evaluation metrics and maintain the existing code quality standards.