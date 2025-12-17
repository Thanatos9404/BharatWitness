# tests/test_pipeline.py
# BharatWitness comprehensive integration tests with synthetic fixtures

import unittest
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval import QueryContext, RetrievalResult
from pipeline.claim_verification import VerificationSummary, ClaimVerificationResult
from pipeline.answer_builder import AnswerBuilder, AnswerEvidence
from pipeline.temporal_engine import TemporalSpan, TemporalEngine
from utils.span_utils import TextSpan
from main import BharatWitnessAPI, AskRequest, DiffRequest


class TestBharatWitnessAPI(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "corpus_root": "test_corpus",
            "paths": {
                "data_root": "data",
                "processed_root": "data/processed",
                "index_root": "data/processed/index",
                "logs_root": "logs",
                "models_root": "models"
            },
            "retrieval": {
                "dense_model": "intfloat/multilingual-e5-large",
                "sparse_model": "naver/splade-cocondenser-ensembledistil",
                "top_k": 50,
                "hybrid_alpha": 0.5
            },
            "nli": {
                "model": "microsoft/mdeberta-v3-xsmall",
                "threshold": 0.8,
                "batch_size": 16
            },
            "temporal": {
                "enable_precedence": True,
                "default_as_of_date": None
            },
            "answer": {
                "max_length": 1024,
                "min_evidence_spans": 2,
                "refusal_threshold": 0.3
            },
            "evaluation": {
                "metrics_output": "data/processed/reports"
            },
            "system": {
                "seed": 42,
                "offline_mode": False,
                "log_level": "INFO",
                "num_workers": 4
            }
        }

        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.temp_dir) / "test_config.yaml"

        import yaml
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)

        self.mock_retrieval_results = [
            RetrievalResult(
                chunk_id="test_chunk_1",
                text="Banks must maintain a minimum capital adequacy ratio of 9% under Basel III guidelines.",
                score=0.95,
                dense_score=0.9,
                sparse_score=0.8,
                bm25_score=0.7,
                metadata={"document_type": "regulation", "page_num": 1},
                spans=[{
                    "text": "Banks must maintain a minimum capital adequacy ratio of 9%",
                    "byte_start": 0,
                    "byte_end": 58,
                    "page_num": 1,
                    "section_type": "regulation",
                    "confidence": 0.95
                }],
                confidence=0.95
            ),
            RetrievalResult(
                chunk_id="test_chunk_2",
                text="KYC procedures require identity verification and address proof documentation.",
                score=0.88,
                dense_score=0.85,
                sparse_score=0.75,
                bm25_score=0.65,
                metadata={"document_type": "guideline", "page_num": 2},
                spans=[{
                    "text": "KYC procedures require identity verification and address proof documentation",
                    "byte_start": 100,
                    "byte_end": 176,
                    "page_num": 2,
                    "section_type": "guideline",
                    "confidence": 0.88
                }],
                confidence=0.88
            )
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('pipeline.retrieval.HybridRetriever')
    @patch('pipeline.claim_verification.ClaimVerificationPipeline')
    @patch('pipeline.answer_builder.AnswerBuilder')
    @patch('pipeline.temporal_engine.TemporalEngine')
    @patch('pipeline.index_build.HybridIndexBuilder')
    def test_api_initialization(self, mock_index_builder, mock_temporal, mock_answer_builder, mock_verifier,
                                mock_retriever):
        mock_index_builder.return_value.load_indices.return_value = True

        api = BharatWitnessAPI(str(self.test_config_path))

        self.assertIsNotNone(api.retriever)
        self.assertIsNotNone(api.verifier)
        self.assertIsNotNone(api.answer_builder)
        self.assertIsNotNone(api.temporal_engine)

    @patch('pipeline.retrieval.HybridRetriever')
    @patch('pipeline.claim_verification.ClaimVerificationPipeline')
    @patch('pipeline.answer_builder.AnswerBuilder')
    @patch('pipeline.temporal_engine.TemporalEngine')
    @patch('pipeline.index_build.HybridIndexBuilder')
    async def test_ask_question_success(self, mock_index_builder, mock_temporal, mock_answer_builder, mock_verifier,
                                        mock_retriever):
        mock_index_builder.return_value.load_indices.return_value = True

        mock_retriever.return_value.retrieve.return_value = self.mock_retrieval_results

        mock_temporal_spans = [
            TemporalSpan(
                span_id="test_span_1",
                text="Banks must maintain a minimum capital adequacy ratio of 9%",
                document_id="test_doc_1",
                effective_date=datetime(2022, 1, 1),
                expiry_date=None,
                publication_date=datetime(2021, 12, 15),
                repeal_date=None,
                status="active",
                precedence_weight=8.0,
                metadata={"document_type": "regulation", "confidence": 0.95}
            )
        ]

        mock_temporal.return_value.create_temporal_spans.return_value = mock_temporal_spans
        mock_temporal.return_value.filter_spans_by_date.return_value = (mock_temporal_spans, [])
        mock_temporal.return_value.resolve_conflicts.return_value = (mock_temporal_spans, [])

        mock_verification_summary = VerificationSummary(
            total_claims=1,
            supported_claims=1,
            refuted_claims=0,
            uncertain_claims=0,
            average_confidence=0.95,
            verification_threshold=0.8,
            refusal_recommended=False
        )
        mock_verifier.return_value.verify_answer.return_value = mock_verification_summary

        mock_built_answer = Mock()
        mock_built_answer.answer_text = "Banks must maintain a minimum capital adequacy ratio of 9% under Basel III guidelines."
        mock_built_answer.citations = ["[1] Regulation, Page 1, Bytes 0-58 (Confidence: 0.95)"]
        mock_built_answer.verification_summary = {"total_claims": 1, "supported_claims": 1}
        mock_built_answer.refusal_reason = None

        mock_answer_builder.return_value.build_answer.return_value = mock_built_answer

        api = BharatWitnessAPI(str(self.test_config_path))

        request = AskRequest(
            query="What is the minimum capital adequacy ratio for banks?",
            as_of_date=datetime(2023, 1, 1),
            confidence_threshold=0.5,
            max_results=10
        )

        response = await api.ask_question(request)

        self.assertIsNotNone(response.answer)
        self.assertGreater(len(response.answer), 0)
        self.assertIsNone(response.refusal_reason)
        self.assertGreater(response.processing_time, 0)
        self.assertEqual(len(response.citations), 1)

    @patch('pipeline.retrieval.HybridRetriever')
    @patch('pipeline.claim_verification.ClaimVerificationPipeline')
    @patch('pipeline.answer_builder.AnswerBuilder')
    @patch('pipeline.temporal_engine.TemporalEngine')
    @patch('pipeline.index_build.HybridIndexBuilder')
    async def test_ask_question_no_results(self, mock_index_builder, mock_temporal, mock_answer_builder, mock_verifier,
                                           mock_retriever):
        mock_index_builder.return_value.load_indices.return_value = True
        mock_retriever.return_value.retrieve.return_value = []

        api = BharatWitnessAPI(str(self.test_config_path))

        request = AskRequest(
            query="What is an obscure regulation that does not exist?",
            confidence_threshold=0.5,
            max_results=10
        )

        response = await api.ask_question(request)

        self.assertIn("No relevant information found", response.answer)
        self.assertIsNotNone(response.refusal_reason)
        self.assertEqual(len(response.citations), 0)

    @patch('pipeline.retrieval.HybridRetriever')
    @patch('pipeline.claim_verification.ClaimVerificationPipeline')
    @patch('pipeline.answer_builder.AnswerBuilder')
    @patch('pipeline.temporal_engine.TemporalEngine')
    @patch('pipeline.index_build.HybridIndexBuilder')
    async def test_compute_diff(self, mock_index_builder, mock_temporal, mock_answer_builder, mock_verifier,
                                mock_retriever):
        mock_index_builder.return_value.load_indices.return_value = True
        mock_retriever.return_value.retrieve.return_value = self.mock_retrieval_results

        mock_temporal_spans = [
            TemporalSpan(
                span_id="test_span_1",
                text="Old regulation text",
                document_id="test_doc_1",
                effective_date=datetime(2020, 1, 1),
                expiry_date=None,
                publication_date=datetime(2019, 12, 15),
                repeal_date=None,
                status="active",
                precedence_weight=8.0,
                metadata={"document_type": "regulation"}
            )
        ]

        mock_temporal.return_value.create_temporal_spans.return_value = mock_temporal_spans
        mock_temporal.return_value.filter_spans_by_date.return_value = (mock_temporal_spans, [])
        mock_temporal.return_value.resolve_conflicts.return_value = (mock_temporal_spans, [])

        mock_verification_summary = VerificationSummary(
            total_claims=1,
            supported_claims=1,
            refuted_claims=0,
            uncertain_claims=0,
            average_confidence=0.9,
            verification_threshold=0.8,
            refusal_recommended=False
        )
        mock_verifier.return_value.verify_answer.return_value = mock_verification_summary

        mock_old_answer = Mock()
        mock_old_answer.answer_text = "Old banking regulation text"
        mock_old_answer.citations = []
        mock_old_answer.verification_summary = {}
        mock_old_answer.refusal_reason = None

        mock_new_answer = Mock()
        mock_new_answer.answer_text = "New updated banking regulation text"
        mock_new_answer.citations = []
        mock_new_answer.verification_summary = {}
        mock_new_answer.refusal_reason = None

        mock_answer_builder.return_value.build_answer.side_effect = [mock_old_answer, mock_new_answer]

        mock_diff = {
            "text_diff": ["- Old banking regulation text", "+ New updated banking regulation text"],
            "evidence_diff": {"added": ["new_span"], "removed": ["old_span"], "retained": []},
            "metadata_diff": {"hash_changed": True, "old_span_count": 1, "new_span_count": 1}
        }
        mock_answer_builder.return_value.create_versioned_diff.return_value = mock_diff

        api = BharatWitnessAPI(str(self.test_config_path))

        request = DiffRequest(
            query="What are the banking regulations?",
            old_date=datetime(2022, 1, 1),
            new_date=datetime(2023, 1, 1)
        )

        response = await api.compute_diff(request)

        self.assertEqual(response.query, request.query)
        self.assertEqual(response.old_date, request.old_date)
        self.assertEqual(response.new_date, request.new_date)
        self.assertIsNotNone(response.text_diff)
        self.assertIsNotNone(response.evidence_diff)
        self.assertTrue(response.summary["hash_changed"])

    @patch('pipeline.retrieval.HybridRetriever')
    @patch('pipeline.claim_verification.ClaimVerificationPipeline')
    @patch('pipeline.answer_builder.AnswerBuilder')
    @patch('pipeline.temporal_engine.TemporalEngine')
    @patch('pipeline.index_build.HybridIndexBuilder')
    def test_health_check(self, mock_index_builder, mock_temporal, mock_answer_builder, mock_verifier, mock_retriever):
        mock_index_builder.return_value.load_indices.return_value = True
        mock_index_builder.return_value.get_index_stats.return_value = {"total_chunks": 100}
        mock_retriever.return_value.get_retrieval_stats.return_value = {"hybrid_alpha": 0.5}

        api = BharatWitnessAPI(str(self.test_config_path))

        health_response = api.get_health_status()

        self.assertEqual(health_response.status, "healthy")
        self.assertTrue(health_response.system_info["components_initialized"])
        self.assertIsNotNone(health_response.timestamp)
        self.assertIn("total_chunks", health_response.indices_status)


class TestSyntheticFixtures(unittest.TestCase):
    def setUp(self):
        self.synthetic_documents = [
            {
                "id": "rbi_circular_2023_001",
                "title": "RBI Circular on Banking Regulations",
                "content": "All banks must maintain a minimum capital adequacy ratio of 11.5% as per the latest Basel III norms. This revision is effective from January 1, 2023.",
                "effective_date": "2023-01-01",
                "document_type": "circular",
                "authority": "RBI"
            },
            {
                "id": "kyc_guidelines_2022",
                "title": "KYC Guidelines for Financial Institutions",
                "content": "Know Your Customer procedures must include Aadhaar-based identity verification, PAN verification, and address proof. Video-based KYC is now permitted for account opening.",
                "effective_date": "2022-06-01",
                "document_type": "guideline",
                "authority": "RBI"
            },
            {
                "id": "digital_payment_act_2020",
                "title": "Digital Payment and Settlement Act",
                "content": "Digital payment service providers must obtain authorization from RBI. UPI transaction limits are set at Rs. 1 lakh per day for individuals.",
                "effective_date": "2020-03-01",
                "document_type": "act",
                "authority": "Government of India"
            }
        ]

        self.synthetic_qa_pairs = [
            {
                "question": "What is the current capital adequacy ratio requirement for banks?",
                "expected_answer": "Banks must maintain a minimum capital adequacy ratio of 11.5% as per Basel III norms, effective from January 1, 2023.",
                "relevant_documents": ["rbi_circular_2023_001"],
                "as_of_date": "2023-06-01"
            },
            {
                "question": "What are the KYC requirements for financial institutions?",
                "expected_answer": "KYC procedures must include Aadhaar-based identity verification, PAN verification, and address proof. Video-based KYC is permitted.",
                "relevant_documents": ["kyc_guidelines_2022"],
                "as_of_date": "2023-01-01"
            },
            {
                "question": "What is the UPI transaction limit for individuals?",
                "expected_answer": "UPI transaction limits are set at Rs. 1 lakh per day for individuals under the Digital Payment and Settlement Act.",
                "relevant_documents": ["digital_payment_act_2020"],
                "as_of_date": "2023-01-01"
            }
        ]

    def test_synthetic_document_structure(self):
        for doc in self.synthetic_documents:
            self.assertIn("id", doc)
            self.assertIn("title", doc)
            self.assertIn("content", doc)
            self.assertIn("effective_date", doc)
            self.assertIn("document_type", doc)
            self.assertIn("authority", doc)

    def test_synthetic_qa_structure(self):
        for qa in self.synthetic_qa_pairs:
            self.assertIn("question", qa)
            self.assertIn("expected_answer", qa)
            self.assertIn("relevant_documents", qa)
            self.assertIn("as_of_date", qa)

    def test_document_qa_alignment(self):
        doc_ids = {doc["id"] for doc in self.synthetic_documents}

        for qa in self.synthetic_qa_pairs:
            for doc_id in qa["relevant_documents"]:
                self.assertIn(doc_id, doc_ids, f"QA references non-existent document: {doc_id}")


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "corpus_root": str(self.temp_dir / Path("corpus")),
            "paths": {
                "processed_root": str(self.temp_dir / Path("processed")),
                "index_root": str(self.temp_dir / Path("processed/index")),
                "logs_root": str(self.temp_dir / Path("logs")),
                "models_root": str(self.temp_dir / Path("models"))
            },
            "system": {"seed": 42, "log_level": "INFO"},
            "retrieval": {"top_k": 10, "hybrid_alpha": 0.5},
            "nli": {"threshold": 0.8},
            "temporal": {"enable_precedence": True},
            "answer": {"refusal_threshold": 0.3}
        }

        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_config_loading(self):
        import yaml
        with open(self.config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.assertEqual(loaded_config["system"]["seed"], 42)
        self.assertEqual(loaded_config["retrieval"]["hybrid_alpha"], 0.5)

    def test_directory_creation(self):
        for path_key in ["processed_root", "index_root", "logs_root", "models_root"]:
            path = Path(self.test_config["paths"][path_key])
            path.mkdir(parents=True, exist_ok=True)
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
