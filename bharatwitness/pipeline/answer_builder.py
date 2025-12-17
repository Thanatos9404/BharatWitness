# pipeline/answer_builder.py
# BharatWitness answer assembly with constrained decoding and refusal policies

import regex as re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import logging
from datetime import datetime
import hashlib
import difflib

from pipeline.claim_verification import VerificationSummary, ClaimVerificationResult
from pipeline.temporal_engine import TemporalSpan
from utils.span_utils import SpanManager


@dataclass
class AnswerEvidence:
    span_id: str
    text: str
    document_id: str
    byte_range: Tuple[int, int]
    confidence: float
    verification_label: str
    citation_key: str
    metadata: Dict[str, Any]


@dataclass
class BuiltAnswer:
    answer_text: str
    evidence_spans: List[AnswerEvidence]
    citations: List[str]
    verification_summary: Dict[str, Any]
    refusal_reason: Optional[str]
    metadata: Dict[str, Any]
    answer_hash: str


class ConstrainedDecoder:
    def __init__(self, max_length: int = 1024, min_evidence_spans: int = 2):
        self.max_length = max_length
        self.min_evidence_spans = min_evidence_spans
        self.logger = logging.getLogger("bharatwitness.decoder")

    def decode_from_spans(self, verified_spans: List[TemporalSpan],
                          verification_results: List[ClaimVerificationResult]) -> str:
        if not verified_spans:
            return ""

        verified_span_ids = {result.span_range for result in verification_results if result.label == 'supported'}

        valid_texts = []
        for span in verified_spans:
            span_range = (span.metadata.get('original_span', {}).get('byte_start', 0),
                          span.metadata.get('original_span', {}).get('byte_end', 0))

            if span_range in verified_span_ids or not verification_results:
                valid_texts.append(span.text.strip())

        if not valid_texts:
            return ""

        answer_parts = []
        current_length = 0

        for text in valid_texts:
            if current_length + len(text) + 1 > self.max_length:
                break

            if text and not any(self._is_duplicate_content(text, existing) for existing in answer_parts):
                answer_parts.append(text)
                current_length += len(text) + 1

        return self._join_coherently(answer_parts)

    def _is_duplicate_content(self, text1: str, text2: str) -> bool:
        clean1 = re.sub(r'\W+', ' ', text1.lower()).strip()
        clean2 = re.sub(r'\W+', ' ', text2.lower()).strip()

        words1 = set(clean1.split())
        words2 = set(clean2.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1.intersection(words2))
        smaller_set_size = min(len(words1), len(words2))

        return overlap / smaller_set_size > 0.8

    def _join_coherently(self, text_parts: List[str]) -> str:
        if not text_parts:
            return ""

        if len(text_parts) == 1:
            return text_parts[0]

        result = text_parts[0]

        for part in text_parts[1:]:
            if result.endswith('.') or result.endswith('!') or result.endswith('?'):
                result += f" {part}"
            elif result.endswith(',') or result.endswith(';'):
                result += f" {part}"
            else:
                result += f". {part}"

        return result


class RefusalPolicy:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.answer_config = self.config["answer"]
        self.refusal_threshold = self.answer_config["refusal_threshold"]
        self.min_evidence_spans = self.answer_config["min_evidence_spans"]

        self.logger = logging.getLogger("bharatwitness.refusal")

    def should_refuse(self, verification_summary: VerificationSummary, spans: List[TemporalSpan], answer_text: str) -> \
    Tuple[bool, Optional[str]]:
        # Only refuse based on NLI if we have actual refuted claims
        if verification_summary.refuted_claims > 0:
            refuted_ratio = verification_summary.refuted_claims / max(verification_summary.total_claims, 1)
            if refuted_ratio > 0.5:
                return True, f"High contradiction ratio: {refuted_ratio:.2f}"

        # Require at least 1 span
        if len(spans) < 1:
            return True, "No evidence spans available"

        if not answer_text.strip():
            return True, "Generated answer is empty"

        return False, None

    def generate_refusal_message(self, reason: str, query: str) -> str:
        refusal_templates = {
            "insufficient_evidence": "I cannot provide a reliable answer to your question due to insufficient evidence in the available documents.",
            "contradictory_evidence": "The available documents contain contradictory information, making it impossible to provide a definitive answer.",
            "low_confidence": "The evidence found does not meet the confidence threshold required for a reliable answer.",
            "refuted_claims": "The evidence contradicts key aspects of what would be required to answer your question.",
            "empty_answer": "No relevant information could be extracted from the available documents to answer your question.",
            "high_contradiction": "The documents contain conflicting information that cannot be reliably resolved."
        }

        for template_key, template in refusal_templates.items():
            if template_key in reason.lower():
                return f"{template} Please rephrase your question or provide additional context."

        return f"I cannot provide a reliable answer due to: {reason}. Please try rephrasing your question."


class CitationManager:
    def __init__(self):
        self.citation_counter = 1
        self.span_to_citation = {}
        self.logger = logging.getLogger("bharatwitness.citations")

    def generate_citations(self, evidence_spans: List[AnswerEvidence]) -> Tuple[str, List[str]]:
        citation_map = {}
        citation_list = []

        for evidence in evidence_spans:
            if evidence.span_id not in self.span_to_citation:
                citation_key = f"[{self.citation_counter}]"
                self.span_to_citation[evidence.span_id] = citation_key
                self.citation_counter += 1

                citation_entry = self._format_citation(evidence, citation_key)
                citation_list.append(citation_entry)
                citation_map[evidence.span_id] = citation_key
            else:
                citation_map[evidence.span_id] = self.span_to_citation[evidence.span_id]

        return citation_map, citation_list

    def _format_citation(self, evidence: AnswerEvidence, citation_key: str) -> str:
        doc_type = evidence.metadata.get('document_type', 'document')
        page_num = evidence.metadata.get('page_num', 'unknown')

        return f"{citation_key} {doc_type.title()}, Page {page_num}, Bytes {evidence.byte_range[0]}-{evidence.byte_range[15]} (Confidence: {evidence.confidence:.2f})"

    def annotate_answer_with_citations(self, answer_text: str, evidence_spans: List[AnswerEvidence]) -> str:
        citation_map, _ = self.generate_citations(evidence_spans)

        annotated_answer = answer_text

        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        annotated_sentences = []

        for sentence in sentences:
            sentence_citations = set()

            for evidence in evidence_spans:
                if self._sentence_uses_evidence(sentence, evidence):
                    citation_key = citation_map.get(evidence.span_id, "")
                    if citation_key:
                        sentence_citations.add(citation_key)

            if sentence_citations:
                sorted_citations = sorted(sentence_citations, key=lambda x: int(x[1:-1]))
                citation_str = "".join(sorted_citations)
                annotated_sentence = f"{sentence.rstrip('.')} {citation_str}."
            else:
                annotated_sentence = sentence

            annotated_sentences.append(annotated_sentence)

        return " ".join(annotated_sentences)

    def _sentence_uses_evidence(self, sentence: str, evidence: AnswerEvidence) -> bool:
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        evidence_words = set(re.findall(r'\w+', evidence.text.lower()))

        if len(sentence_words) == 0 or len(evidence_words) == 0:
            return False

        overlap = len(sentence_words.intersection(evidence_words))
        return overlap >= min(3, len(sentence_words) * 0.3)


class ContradictionResolver:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.contradictions")

    def suppress_contradictions(self, spans: List[TemporalSpan], verification_results: List[ClaimVerificationResult]) -> \
    Tuple[List[TemporalSpan], List[TemporalSpan]]:
        refuted_ranges = {result.span_range for result in verification_results if result.label == 'refuted'}

        valid_spans = []
        suppressed_spans = []

        for span in spans:
            span_range = (span.metadata.get('original_span', {}).get('byte_start', 0),
                          span.metadata.get('original_span', {}).get('byte_end', 0))

            if span_range in refuted_ranges:
                suppressed_spans.append(span)
            else:
                valid_spans.append(span)

        self.logger.info(
            f"Suppressed {len(suppressed_spans)} contradictory spans, retained {len(valid_spans)} valid spans")
        return valid_spans, suppressed_spans

    def resolve_precedence_conflicts(self, spans: List[TemporalSpan]) -> List[TemporalSpan]:
        if len(spans) <= 1:
            return spans

        conflict_groups = self._identify_conflict_groups(spans)
        resolved_spans = []

        for group in conflict_groups:
            if len(group) == 1:
                resolved_spans.extend(group)
            else:
                winner = max(group, key=lambda s: (s.precedence_weight, s.effective_date or datetime.min))
                resolved_spans.append(winner)
                self.logger.debug(f"Resolved conflict: chose {winner.span_id} over {len(group) - 1} alternatives")

        return resolved_spans

    def _identify_conflict_groups(self, spans: List[TemporalSpan]) -> List[List[TemporalSpan]]:
        conflict_groups = []
        processed = set()

        for i, span1 in enumerate(spans):
            if span1.span_id in processed:
                continue

            conflict_group = [span1]
            processed.add(span1.span_id)

            for j, span2 in enumerate(spans[i + 1:], i + 1):
                if span2.span_id in processed:
                    continue

                if self._spans_conflict(span1, span2):
                    conflict_group.append(span2)
                    processed.add(span2.span_id)

            conflict_groups.append(conflict_group)

        return conflict_groups

    def _spans_conflict(self, span1: TemporalSpan, span2: TemporalSpan) -> bool:
        text1_clean = re.sub(r'\W+', ' ', span1.text.lower()).strip()
        text2_clean = re.sub(r'\W+', ' ', span2.text.lower()).strip()

        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))

        return similarity > 0.7


class AnswerBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.answer_config = self.config["answer"]

        self.decoder = ConstrainedDecoder(
            max_length=self.answer_config["max_length"],
            min_evidence_spans=self.answer_config["min_evidence_spans"]
        )

        self.refusal_policy = RefusalPolicy(config_path)
        self.citation_manager = CitationManager()
        self.contradiction_resolver = ContradictionResolver()

        self.span_manager = SpanManager()
        self.logger = logging.getLogger("bharatwitness.answer_builder")

    def build_answer(self, query: str, verified_spans: List[TemporalSpan], verification_summary: VerificationSummary,
                     verification_results: List[ClaimVerificationResult]) -> BuiltAnswer:

        valid_spans, suppressed_spans = self.contradiction_resolver.suppress_contradictions(verified_spans,
                                                                                            verification_results)

        resolved_spans = self.contradiction_resolver.resolve_precedence_conflicts(valid_spans)

        should_refuse, refusal_reason = self.refusal_policy.should_refuse(verification_summary, resolved_spans, "")

        if should_refuse:
            refusal_message = self.refusal_policy.generate_refusal_message(refusal_reason, query)

            return BuiltAnswer(
                answer_text=refusal_message,
                evidence_spans=[],
                citations=[],
                verification_summary=verification_summary.__dict__,
                refusal_reason=refusal_reason,
                metadata={
                    'total_original_spans': len(verified_spans),
                    'suppressed_spans': len(suppressed_spans),
                    'refused': True
                },
                answer_hash=self._compute_answer_hash(refusal_message)
            )

        answer_text = self.decoder.decode_from_spans(resolved_spans, verification_results)

        should_refuse_final, final_refusal_reason = self.refusal_policy.should_refuse(verification_summary,
                                                                                      resolved_spans, answer_text)

        if should_refuse_final:
            refusal_message = self.refusal_policy.generate_refusal_message(final_refusal_reason, query)

            return BuiltAnswer(
                answer_text=refusal_message,
                evidence_spans=[],
                citations=[],
                verification_summary=verification_summary.__dict__,
                refusal_reason=final_refusal_reason,
                metadata={
                    'total_original_spans': len(verified_spans),
                    'suppressed_spans': len(suppressed_spans),
                    'refused': True
                },
                answer_hash=self._compute_answer_hash(refusal_message)
            )

        evidence_spans = self._create_evidence_spans(resolved_spans, verification_results)

        annotated_answer = self.citation_manager.annotate_answer_with_citations(answer_text, evidence_spans)

        _, citation_list = self.citation_manager.generate_citations(evidence_spans)

        return BuiltAnswer(
            answer_text=annotated_answer,
            evidence_spans=evidence_spans,
            citations=citation_list,
            verification_summary=verification_summary.__dict__,
            refusal_reason=None,
            metadata={
                'total_original_spans': len(verified_spans),
                'suppressed_spans': len(suppressed_spans),
                'final_spans': len(resolved_spans),
                'refused': False
            },
            answer_hash=self._compute_answer_hash(annotated_answer)
        )

    def _create_evidence_spans(self, spans: List[TemporalSpan], verification_results: List[ClaimVerificationResult]) -> \
    List[AnswerEvidence]:
        verification_map = {result.span_range: result for result in verification_results}

        evidence_spans = []

        for span in spans:
            span_range = (span.metadata.get('original_span', {}).get('byte_start', 0),
                          span.metadata.get('original_span', {}).get('byte_end', 0))

            verification_result = verification_map.get(span_range)
            verification_label = verification_result.label if verification_result else 'uncertain'

            evidence = AnswerEvidence(
                span_id=span.span_id,
                text=span.text,
                document_id=span.document_id,
                byte_range=span_range,
                confidence=span.metadata.get('confidence', 0.0),
                verification_label=verification_label,
                citation_key="",
                metadata=span.metadata
            )

            evidence_spans.append(evidence)

        return evidence_spans

    def _compute_answer_hash(self, answer_text: str) -> str:
        return hashlib.sha256(answer_text.encode('utf-8')).hexdigest()[:16]

    def create_versioned_diff(self, old_answer: BuiltAnswer, new_answer: BuiltAnswer) -> Dict[str, Any]:
        text_diff = list(difflib.unified_diff(
            old_answer.answer_text.splitlines(keepends=True),
            new_answer.answer_text.splitlines(keepends=True),
            fromfile="old_answer",
            tofile="new_answer"
        ))

        evidence_diff = self._compute_evidence_diff(old_answer.evidence_spans, new_answer.evidence_spans)

        return {
            'text_diff': text_diff,
            'evidence_diff': evidence_diff,
            'metadata_diff': {
                'old_hash': old_answer.answer_hash,
                'new_hash': new_answer.answer_hash,
                'old_span_count': len(old_answer.evidence_spans),
                'new_span_count': len(new_answer.evidence_spans),
                'hash_changed': old_answer.answer_hash != new_answer.answer_hash
            }
        }

    def _compute_evidence_diff(self, old_evidence: List[AnswerEvidence], new_evidence: List[AnswerEvidence]) -> Dict[
        str, List]:
        old_ids = {ev.span_id for ev in old_evidence}
        new_ids = {ev.span_id for ev in new_evidence}

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids
        common_ids = old_ids & new_ids

        return {
            'added': [ev.span_id for ev in new_evidence if ev.span_id in added_ids],
            'removed': [ev.span_id for ev in old_evidence if ev.span_id in removed_ids],
            'retained': list(common_ids)
        }
