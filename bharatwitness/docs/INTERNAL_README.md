# BharatWitness Internal Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BharatWitness RAG System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   FastAPI   │───▶│  Retrieval  │───▶│  Temporal   │───▶│   Answer    │  │
│  │   Server    │    │  Pipeline   │    │   Engine    │    │   Builder   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Health    │    │   FAISS +   │    │ Precedence  │    │    Claim    │  │
│  │   Check     │    │ BM25 + TF-IDF│   │    Graph    │    │ Verification│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Entrypoints

### API Server
```bash
python main.py --host 127.0.0.1 --port 8000
```

### End-to-End Pipeline
```bash
python scripts/run_end_to_end.py --config config/config.yaml
```

### Index Building Only
```bash
python build_indices.py
```

### Evaluation Only
```bash
python scripts/evaluate.py --qa-data data/gold_qa/test.json
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | System health status |
| `/ask` | POST | Query with temporal filtering |
| `/diff` | POST | Compare answers across dates |
| `/docs` | GET | OpenAPI documentation |

---

## Query Flow

1. **Request** → `main.py:ask_question()`
2. **Retrieval** → `pipeline/retrieval.py:HybridRetriever.retrieve()`
   - Dense search (FAISS HNSW or TF-IDF fallback)
   - Sparse search (SPLADE or TF-IDF)
   - BM25 search
   - Score fusion and ranking
3. **Temporal Filtering** → `pipeline/temporal_engine.py`
   - Date-based span filtering
   - Precedence resolution
4. **Verification** → `pipeline/claim_verification.py`
   - Claim extraction
   - NLI-based entailment check
5. **Answer Building** → `pipeline/answer_builder.py`
   - Constrained decoding
   - Citation generation
   - Refusal policy check

---

## Configuration

Core settings in `config/config.yaml`:

- **OCR**: trust_threshold, languages (en, hi)
- **Retrieval**: dense_model (multilingual-e5), top_k, hybrid_alpha
- **NLI**: model (mDeBERTa), threshold, batch_size
- **Temporal**: enable_precedence, as_of_date
- **Answer**: max_length, min_evidence_spans, refusal_threshold

---

## Directory Structure

```
bharatwitness/
├── config/          # YAML configuration
├── data/            # Corpus and processed data
├── evaluation/      # Metrics and ablation studies
├── models/          # Trained model artifacts
├── ocr/             # PaddleOCR pipeline
├── pipeline/        # Core RAG components
├── scripts/         # CLI tools
├── tests/           # pytest test suite
└── utils/           # Shared utilities
```

---

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | ≥95% | Attributable claims ratio |
| Contradiction Rate | ≤2% | Refuted claims ratio |
| nDCG@10 | ≥0.85 | Retrieval quality |
| Latency p95 | ≤3.0s | Response time |
| Temporal Accuracy | ≥98% | Date-valid spans |
| Robustness Drop | ≤5% | Code-mixed query F1 |
