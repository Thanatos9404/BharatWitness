# pipeline/claim_verification.py
# BharatWitness NLI-based claim verification with confidence calibration

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
import regex as re
import json
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
import pickle

from utils.span_utils import TextSpan


@dataclass
class ClaimVerificationResult:
    claim: str
    evidence: str
    label: str
    confidence: float
    raw_scores: Dict[str, float]
    calibrated_confidence: float
    span_range: Tuple[int, int]


@dataclass
class VerificationSummary:
    total_claims: int
    supported_claims: int
    refuted_claims: int
    uncertain_claims: int
    average_confidence: float
    verification_threshold: float
    refusal_recommended: bool


class ClaimExtractor:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.claims")

        self.claim_patterns = [
            r'(?:According to|As per|Under|In accordance with).*?(?:\.|;|$)',
            r'(?:The|This|Such).*?(?:shall|must|may|will|can).*?(?:\.|;|$)',
            r'(?:Every|Any|All|No).*?(?:person|entity|organization).*?(?:\.|;|$)',
            r'(?:It is|This is).*?(?:mandatory|required|prohibited|permitted).*?(?:\.|;|$)',
            r'(?:The rate|The amount|The limit).*?(?:is|shall be|will be).*?(?:\.|;|$)'
        ]

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        claims = []
        sentences = self._split_into_sentences(text)

        for i, sentence in enumerate(sentences):
            if self._is_factual_claim(sentence):
                claim_start = text.find(sentence)
                claim_end = claim_start + len(sentence)

                claims.append({
                    'text': sentence.strip(),
                    'start': claim_start,
                    'end': claim_end,
                    'sentence_index': i,
                    'claim_type': self._classify_claim_type(sentence)
                })

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_factual_claim(self, sentence: str) -> bool:
        if len(sentence.split()) < 5:
            return False

        factual_indicators = [
            'shall', 'must', 'will', 'is', 'are', 'was', 'were',
            'according to', 'as per', 'under', 'pursuant to',
            'provided that', 'subject to', 'in case of'
        ]

        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)

    def _classify_claim_type(self, sentence: str) -> str:
        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ['shall', 'must', 'required']):
            return 'obligation'
        elif any(word in sentence_lower for word in ['may', 'can', 'permitted']):
            return 'permission'
        elif any(word in sentence_lower for word in ['prohibited', 'not', 'shall not']):
            return 'prohibition'
        elif any(word in sentence_lower for word in ['rate', 'amount', 'percentage', '%']):
            return 'quantitative'
        else:
            return 'descriptive'


class NLIVerifier:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.nli_config = self.config["nli"]
        self.model_name = self.nli_config["model"]
        self.threshold = self.nli_config["threshold"]
        self.batch_size = self.nli_config["batch_size"]

        self.models_root = Path(self.config["paths"]["models_root"])
        self.nli_model_path = self.models_root / "nli"

        self.logger = logging.getLogger("bharatwitness.nli")

        self.tokenizer = None
        self.model = None
        self.calibrator = None

        self.label_mapping = {
            0: 'entailment',
            1: 'neutral',
            2: 'contradiction'
        }

        self.verification_mapping = {
            'entailment': 'supported',
            'neutral': 'uncertain',
            'contradiction': 'refuted'
        }

        self._load_model()

    def _load_model(self):
        try:
            custom_model_path = self.nli_model_path / "pytorch_model.bin"
            if custom_model_path.exists():
                self.logger.info(f"Loading custom NLI model from {self.nli_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.nli_model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.nli_model_path))
            else:
                self.logger.info(f"Loading pre-trained NLI model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.logger.info("NLI model moved to GPU")

            self.model.eval()

            calibrator_path = self.nli_model_path / "calibrator.pkl"
            if calibrator_path.exists():
                with open(calibrator_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
                self.logger.info("Loaded confidence calibrator")

        except Exception as e:
            self.logger.error(f"Failed to load NLI model: {e}")
            raise

    def verify_claims(self, claims: List[Dict[str, Any]], evidence_spans: List[TextSpan]) -> List[
        ClaimVerificationResult]:
        verification_results = []

        evidence_texts = [span.text for span in evidence_spans]

        for claim_data in claims:
            claim_text = claim_data['text']

            best_evidence = self._find_best_evidence(claim_text, evidence_texts)

            if best_evidence:
                result = self._verify_single_claim(claim_text, best_evidence, claim_data)
                verification_results.append(result)
            else:
                result = ClaimVerificationResult(
                    claim=claim_text,
                    evidence="",
                    label="uncertain",
                    confidence=0.0,
                    raw_scores={},
                    calibrated_confidence=0.0,
                    span_range=(claim_data['start'], claim_data['end'])
                )
                verification_results.append(result)

        return verification_results

    def _find_best_evidence(self, claim: str, evidence_texts: List[str]) -> Optional[str]:
        if not evidence_texts:
            return None

        from sentence_transformers import SentenceTransformer, util

        try:
            encoder = SentenceTransformer('all-MiniLM-L6-v2')

            claim_embedding = encoder.encode([claim])
            evidence_embeddings = encoder.encode(evidence_texts)

            similarities = util.cos_sim(claim_embedding, evidence_embeddings)[0]
            best_idx = similarities.argmax().item()

            if similarities[best_idx] > 0.3:
                return evidence_texts[best_idx]

        except Exception as e:
            self.logger.warning(f"Similarity search failed: {e}")

        return evidence_texts[0] if evidence_texts else None

    def _verify_single_claim(self, claim: str, evidence: str, claim_data: Dict[str, Any]) -> ClaimVerificationResult:
        try:
            inputs = self.tokenizer(
                claim,
                evidence,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted_label_idx = np.argmax(probabilities)

                nli_label = self.label_mapping[predicted_label_idx]
                verification_label = self.verification_mapping[nli_label]

                raw_confidence = float(probabilities[predicted_label_idx])

                raw_scores = {
                    self.label_mapping[i]: float(probabilities[i])
                    for i in range(len(probabilities))
                }

                calibrated_confidence = self._calibrate_confidence(probabilities, predicted_label_idx)

                return ClaimVerificationResult(
                    claim=claim,
                    evidence=evidence,
                    label=verification_label,
                    confidence=raw_confidence,
                    raw_scores=raw_scores,
                    calibrated_confidence=calibrated_confidence,
                    span_range=(claim_data['start'], claim_data['end'])
                )

        except Exception as e:
            self.logger.error(f"Claim verification failed: {e}")
            return ClaimVerificationResult(
                claim=claim,
                evidence=evidence,
                label="uncertain",
                confidence=0.0,
                raw_scores={},
                calibrated_confidence=0.0,
                span_range=(claim_data['start'], claim_data['end'])
            )

    def _calibrate_confidence(self, probabilities: np.ndarray, predicted_idx: int) -> float:
        if self.calibrator is None:
            return float(probabilities[predicted_idx])

        try:
            calibrated_probs = self.calibrator.predict_proba([probabilities])
            return float(calibrated_probs[0][predicted_idx])
        except Exception:
            return float(probabilities[predicted_idx])


class ClaimVerificationPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.extractor = ClaimExtractor()
        self.verifier = NLIVerifier(config_path)
        self.logger = logging.getLogger("bharatwitness.verification")

        self.verification_threshold = self.config["nli"]["threshold"]

    def verify_answer(self, answer_text: str, evidence_spans: List[TextSpan]) -> VerificationSummary:
        claims = self.extractor.extract_claims(answer_text)

        if not claims:
            return VerificationSummary(
                total_claims=0,
                supported_claims=0,
                refuted_claims=0,
                uncertain_claims=0,
                average_confidence=0.0,
                verification_threshold=self.verification_threshold,
                refusal_recommended=True
            )

        verification_results = self.verifier.verify_claims(claims, evidence_spans)

        return self._summarize_verification(verification_results)

    def filter_verified_claims(self, answer_text: str, evidence_spans: List[TextSpan]) -> Tuple[
        str, VerificationSummary]:
        claims = self.extractor.extract_claims(answer_text)
        verification_results = self.verifier.verify_claims(claims, evidence_spans)

        filtered_text = answer_text
        offset = 0

        for result in sorted(verification_results, key=lambda x: x.span_range[0], reverse=True):
            if result.label == 'refuted' or (
                    result.label == 'uncertain' and result.calibrated_confidence < self.verification_threshold):
                start, end = result.span_range
                start -= offset
                end -= offset

                filtered_text = filtered_text[:start] + filtered_text[end:]
                offset += (end - start)

        summary = self._summarize_verification(verification_results)

        return filtered_text.strip(), summary

    def _summarize_verification(self, results: List[ClaimVerificationResult]) -> VerificationSummary:
        if not results:
            return VerificationSummary(
                total_claims=0,
                supported_claims=0,
                refuted_claims=0,
                uncertain_claims=0,
                average_confidence=0.0,
                verification_threshold=self.verification_threshold,
                refusal_recommended=True
            )

        supported = sum(1 for r in results if r.label == 'supported')
        refuted = sum(1 for r in results if r.label == 'refuted')
        uncertain = sum(1 for r in results if r.label == 'uncertain')

        avg_confidence = np.mean([r.calibrated_confidence for r in results])

        refusal_recommended = (
                refuted > 0 or
                uncertain / len(results) > 0.5 or
                avg_confidence < self.verification_threshold
        )

        return VerificationSummary(
            total_claims=len(results),
            supported_claims=supported,
            refuted_claims=refuted,
            uncertain_claims=uncertain,
            average_confidence=float(avg_confidence),
            verification_threshold=self.verification_threshold,
            refusal_recommended=refusal_recommended
        )

    def get_detailed_verification_report(self, answer_text: str, evidence_spans: List[TextSpan]) -> Dict[str, Any]:
        claims = self.extractor.extract_claims(answer_text)
        verification_results = self.verifier.verify_claims(claims, evidence_spans)
        summary = self._summarize_verification(verification_results)

        return {
            'summary': summary.__dict__,
            'detailed_results': [
                {
                    'claim': result.claim,
                    'evidence': result.evidence[:200] + "..." if len(result.evidence) > 200 else result.evidence,
                    'label': result.label,
                    'confidence': result.confidence,
                    'calibrated_confidence': result.calibrated_confidence,
                    'raw_scores': result.raw_scores
                }
                for result in verification_results
            ],
            'verification_config': {
                'threshold': self.verification_threshold,
                'model': self.verifier.model_name
            }
        }
