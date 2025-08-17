# tests/test_pipeline.py
# BharatWitness integration tests with synthetic fixtures

import unittest
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import Mock, patch

from bharatwitness.pipeline.retrieval import QueryContext, RetrievalResult
from bharatwitness.pipeline.claim_verification import VerificationSummary, ClaimVerificationResult
from bharatwitness.pipeline.answer_builder import AnswerBuilder, AnswerEvidence
from bharatwitness.pipeline.temporal_engine import TemporalSpan, TemporalEngine
from bharatwitness.utils.span_utils import TextSpan


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.test_config_path = "config/config.yaml"
        self.answer_builder = AnswerBuilder(self.test_config_path)

        self.sample_spans = self._create_sample_temporal_spans()
        self.sample_verification_summary = self._create_sample_verification_summary()
        self.sample_verification_results = self._create_sample_verification_results()

    def _create_sample_temporal_spans(self):
        return [
            TemporalSpan(
                span_id="doc1_section1",
                text="Banks must maintain minimum capital adequacy ratio of 9% as per RBI guidelines.",
                document_id="doc1",
                effective_date=datetime(2022, 1, 1),
                expiry_date=None,
                publication_date=datetime(2021, 12, 15),
                repeal_date=None,
                supersedes=[],
                superseded_by=None,
                status="active",
                precedence_weight=8.0,
                metadata={
                    'document_type': 'regulation',
                    'confidence': 0.95,
                    'original_span': {'byte_start': 100, 'byte_end': 200},
                    'page_num': 1
                }
            ),
            TemporalSpan(
                span_id="doc2_section1",
                text="KYC requirements include identity verification and address proof documentation.",
                document_id="doc2",
                effective_date=datetime(2021, 6, 1),
                expiry_date=None,
                publication_date=datetime(2021, 5, 15),
                repeal_date=None,
                supersedes=[],
                superseded_by=None,
                status="active",
                precedence_weight=7.0,
                metadata={
                    'document_type': 'guideline',
                    'confidence': 0.88,
                    'original_span': {'byte_start': 300, 'byte_end': 400},
                    'page_num': 2
                }
            )
        ]

    def _create_sample_verification_summary(self):
        return VerificationSummary(
            total_claims=2,
            supported_claims=2,
            refuted_claims=0,
            uncertain_claims=0,
            average_confidence=0.92,
            verification_threshold=0.8,
            refusal_recommended=False
        )

    def _create_sample_verification_results(self):
        return [
            ClaimVerificationResult(
                claim="Banks must maintain minimum capital adequacy ratio of 9%",
                evidence="RBI guidelines specify capital requirements",
                label="supported",
                confidence=0.95,
                raw_scores={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
                calibrated_confidence=0.93,
                span_range=(100, 200)
            ),
            ClaimVerificationResult(
                claim="KYC requirements include identity verification",
                evidence="Banking regulations specify KYC procedures",
                label="supported",
                confidence=0.88,
                raw_scores={"entailment": 0.88, "neutral": 0.08, "contradiction": 0.04},
                calibrated_confidence=0.86,
                span_range=(300, 400)
            )
        ]

    def test_successful_answer_building(self):
        query = "What are the capital adequacy and KYC requirements for banks?"

        result = self.answer_builder.build_answer(
            query=query,
            verified_spans=self.sample_spans,
            verification_summary=self.sample_verification_summary,
            verification_results=self.sample_verification_results
        )

        self.assertIsNotNone(result.answer_text)
        self.assertGreater(len(result.answer_text), 0)
        self.assertEqual(len(result.evidence_spans), 2)
        self.assertIsNone(result.refusal_reason)
        self.assertFalse(result.metadata['refused'])
        self.assertGreater(len(result.citations), 0)

    def test_refusal_due_to_low_confidence(self):
        low_confidence_summary = VerificationSummary(
            total_claims=2,
            supported_claims=1,
            refuted_claims=0,
            uncertain_claims=1,
            average_confidence=0.2,
            verification_threshold=0.8,
            refusal_recommended=True
        )

        query = "What are unclear banking requirements?"

        result = self.answer_builder.build_answer(
            query=query,
            verified_spans=self.sample_spans,
            verification_summary=low_confidence_summary,
            verification_results=self.sample_verification_results
        )

        self.assertIn("cannot provide a reliable answer", result.answer_text.lower())
        self.assertEqual(len(result.evidence_spans), 0)
        self.assertIsNotNone(result.refusal_reason)
        self.assertTrue(result.metadata['refused'])

    def test_contradiction_suppression(self):
        conflicted_verification_results = self.sample_verification_results + [
            ClaimVerificationResult(
                claim="Banks do not need capital adequacy ratios",
                evidence="Contradictory claim",
                label="refuted",
                confidence=0.85,
                raw_scores={"entailment": 0.1, "neutral": 0.05, "contradiction": 0.85},
                calibrated_confidence=0.83,
                span_range=(500, 600)
            )
        ]

        contradicted_span = TemporalSpan(
            span_id="doc3_section1",
            text="Banks do not need capital adequacy ratios according to some interpretations.",
            document_id="doc3",
            effective_date=datetime(2020, 1, 1),
            expiry_date=None,
            publication_date=datetime(2019, 12, 15),
            repeal_date=None,
            supersedes=[],
            superseded_by=None,
            status="active",
            precedence_weight=5.0,
            metadata={
                'document_type': 'opinion',
                'confidence': 0.7,
                'original_span': {'byte_start': 500, 'byte_end': 600},
                'page_num': 3
            }
        )

        spans_with_contradiction = self.sample_spans + [contradicted_span]

        query = "What are banking capital requirements?"

        result = self.answer_builder.build_answer(
            query=query,
            verified_spans=spans_with_contradiction,
            verification_summary=self.sample_verification_summary,
            verification_results=conflicted_verification_results
        )

        self.assertEqual(len(result.evidence_spans), 2)
        self.assertEqual(result.metadata['suppressed_spans'], 1)

    def test_citation_generation(self):
        query = "What are banking requirements?"

        result = self.answer_builder.build_answer(
            query=query,
            verified_spans=self.sample_spans,
            verification_summary=self.sample_verification_summary,
            verification_results=self.sample_verification_results
        )

        self.assertGreater(len(result.citations), 0)
        self.assertIn("[1]", result.answer_text or "[16]" in result.answer_text)

        for citation in result.citations:
            self.assertIn("Confidence:", citation)
            self.assertIn("Bytes", citation)

    def test_empty_spans_refusal(self):
        query = "What are the requirements?"

        result = self.answer_builder.build_answer(
            query=query,
            verified_spans=[],
            verification_summary=VerificationSummary(
                total_claims=0, supported_claims=0, refuted_claims=0,
                uncertain_claims=0, average_confidence=0.0,
                verification_threshold=0.8, refusal_recommended=True
            ),
            verification_results=[]
        )

        self.assertIsNotNone(result.refusal_reason)
        self.assertTrue(result.metadata['refused'])
        self.assertEqual(len(result.evidence_spans), 0)

    def test_versioned_diff_creation(self):
        query = "Banking requirements"

        old_answer = self.answer_builder.build_answer(
            query=query,
            verified_spans=self.sample_spans[:1],
            verification_summary=self.sample_verification_summary,
            verification_results=self.sample_verification_results[:1]
        )

        new_answer = self.answer_builder.build_answer(
            query=query,
            verified_spans=self.sample_spans,
            verification_summary=self.sample_verification_summary,
            verification_results=self.sample_verification_results
        )

        diff = self.answer_builder.create_versioned_diff(old_answer, new_answer)

        self.assertIn('text_diff', diff)
        self.assertIn('evidence_diff', diff)
        self.assertIn('metadata_diff', diff)

        self.assertEqual(len(diff['evidence_diff']['added']), 1)
        self.assertEqual(len(diff['evidence_diff']['removed']), 0)


class TestAnswerBuilderComponents(unittest.TestCase):
    def setUp(self):
        self.answer_builder = AnswerBuilder()

    def test_constrained_decoder(self):
        spans = [
            TemporalSpan(
                span_id="test1",
                text="First regulation text.",
                document_id="doc1",
                effective_date=None, expiry_date=None,
                publication_date=None, repeal_date=None,
                status="active", precedence_weight=1.0,
                metadata={'original_span': {'byte_start': 0, 'byte_end': 100}}
            ),
            TemporalSpan(
                span_id="test2",
                text="Second regulation text.",
                document_id="doc2",
                effective_date=None, expiry_date=None,
                publication_date=None, repeal_date=None,
                status="active", precedence_weight=1.0,
                metadata={'original_span': {'byte_start': 100, 'byte_end': 200}}
            )
        ]

        verification_results = [
            ClaimVerificationResult(
                claim="Test", evidence="Test", label="supported",
                confidence=0.9, raw_scores={}, calibrated_confidence=0.85,
                span_range=(0, 100)
            ),
            ClaimVerificationResult(
                claim="Test", evidence="Test", label="supported",
                confidence=0.8, raw_scores={}, calibrated_confidence=0.75,
                span_range=(100, 200)
            )
        ]

        result = self.answer_builder.decoder.decode_from_spans(spans, verification_results)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_refusal_policy_logic(self):
        high_confidence_summary = VerificationSummary(
            total_claims=3, supported_claims=3, refuted_claims=0,
            uncertain_claims=0, average_confidence=0.95,
            verification_threshold=0.8, refusal_recommended=False
        )

        should_refuse, reason = self.answer_builder.refusal_policy.should_refuse(
            high_confidence_summary, [Mock()], "Valid answer text with content."
        )

        self.assertFalse(should_refuse)
        self.assertIsNone(reason)

        low_confidence_summary = VerificationSummary(
            total_claims=2, supported_claims=1, refuted_claims=1,
            uncertain_claims=0, average_confidence=0.4,
            verification_threshold=0.8, refusal_recommended=True
        )

        should_refuse, reason = self.answer_builder.refusal_policy.should_refuse(
            low_confidence_summary, [Mock()], "Some answer"
        )

        self.assertTrue(should_refuse)
        self.assertIsNotNone(reason)


if __name__ == '__main__':
    unittest.main()
