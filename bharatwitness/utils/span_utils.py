# utils/span_utils.py
# BharatWitness span manipulation and provenance utilities

from typing import List, Dict, Any, Tuple, Optional, Set
import regex as re
from dataclasses import dataclass
import hashlib


@dataclass
class TextSpan:
    text: str
    start: int
    end: int
    document_id: str
    page_num: int
    section_id: str
    confidence: float
    metadata: Dict[str, Any]

    def __hash__(self) -> int:
        return hash((self.text, self.start, self.end, self.document_id))

    def overlaps_with(self, other: 'TextSpan') -> bool:
        return not (self.end <= other.start or other.end <= self.start)

    def merge_with(self, other: 'TextSpan') -> 'TextSpan':
        if not self.overlaps_with(other):
            raise ValueError("Cannot merge non-overlapping spans")

        merged_start = min(self.start, other.start)
        merged_end = max(self.end, other.end)
        merged_text = f"{self.text} {other.text}".strip()

        return TextSpan(
            text=merged_text,
            start=merged_start,
            end=merged_end,
            document_id=self.document_id,
            page_num=min(self.page_num, other.page_num),
            section_id=f"{self.section_id}+{other.section_id}",
            confidence=min(self.confidence, other.confidence),
            metadata={**self.metadata, **other.metadata}
        )


class SpanManager:
    def __init__(self):
        self.span_cache = {}

    def create_span(
            self,
            text: str,
            start: int,
            end: int,
            document_id: str,
            page_num: int,
            section_id: str,
            confidence: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None
    ) -> TextSpan:

        if metadata is None:
            metadata = {}

        return TextSpan(
            text=text,
            start=start,
            end=end,
            document_id=document_id,
            page_num=page_num,
            section_id=section_id,
            confidence=confidence,
            metadata=metadata
        )

    def find_overlapping_spans(self, spans: List[TextSpan], target_span: TextSpan) -> List[TextSpan]:
        overlapping = []
        for span in spans:
            if span.document_id == target_span.document_id and span.overlaps_with(target_span):
                overlapping.append(span)
        return overlapping

    def merge_adjacent_spans(self, spans: List[TextSpan], max_gap: int = 5) -> List[TextSpan]:
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: (s.document_id, s.start))
        merged = [sorted_spans[0]]

        for current_span in sorted_spans[1:]:
            last_merged = merged[-1]

            if (current_span.document_id == last_merged.document_id and
                    current_span.start - last_merged.end <= max_gap):

                try:
                    merged_span = last_merged.merge_with(current_span)
                    merged[-1] = merged_span
                except ValueError:
                    merged.append(current_span)
            else:
                merged.append(current_span)

        return merged

    def filter_spans_by_confidence(self, spans: List[TextSpan], min_confidence: float) -> List[TextSpan]:
        return [span for span in spans if span.confidence >= min_confidence]

    def group_spans_by_document(self, spans: List[TextSpan]) -> Dict[str, List[TextSpan]]:
        grouped = {}
        for span in spans:
            if span.document_id not in grouped:
                grouped[span.document_id] = []
            grouped[span.document_id].append(span)
        return grouped

    def extract_citations(self, spans: List[TextSpan]) -> List[Dict[str, Any]]:
        citations = []

        for span in spans:
            citation = {
                'text': span.text,
                'document_id': span.document_id,
                'page_num': span.page_num,
                'section_id': span.section_id,
                'byte_range': f"{span.start}-{span.end}",
                'confidence': span.confidence,
                'source_hash': self._generate_span_hash(span)
            }
            citations.append(citation)

        return citations

    def _generate_span_hash(self, span: TextSpan) -> str:
        content = f"{span.document_id}:{span.start}:{span.end}:{span.text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def validate_span_integrity(self, spans: List[TextSpan]) -> Dict[str, Any]:
        validation_report = {
            'total_spans': len(spans),
            'valid_spans': 0,
            'invalid_spans': [],
            'duplicate_spans': [],
            'orphaned_spans': []
        }

        seen_spans = set()

        for i, span in enumerate(spans):
            is_valid = True

            if span.start < 0 or span.end <= span.start:
                validation_report['invalid_spans'].append(i)
                is_valid = False

            if not span.text.strip():
                validation_report['invalid_spans'].append(i)
                is_valid = False

            span_signature = (span.text, span.start, span.end, span.document_id)
            if span_signature in seen_spans:
                validation_report['duplicate_spans'].append(i)
            else:
                seen_spans.add(span_signature)

            if not span.document_id or not span.section_id:
                validation_report['orphaned_spans'].append(i)

            if is_valid:
                validation_report['valid_spans'] += 1

        return validation_report

    def resolve_span_conflicts(self, spans: List[TextSpan]) -> List[TextSpan]:
        conflicts = []
        resolved_spans = []

        for i, span1 in enumerate(spans):
            for j, span2 in enumerate(spans[i + 1:], i + 1):
                if span1.overlaps_with(span2) and span1.document_id == span2.document_id:
                    conflicts.append((i, j))

        conflict_indices = set()
        for conf in conflicts:
            conflict_indices.update(conf)

        for i, span in enumerate(spans):
            if i not in conflict_indices:
                resolved_spans.append(span)
            elif span.confidence >= 0.8:
                resolved_spans.append(span)

        return resolved_spans

    def get_span_context(self, target_span: TextSpan, all_spans: List[TextSpan], context_window: int = 200) -> Dict[
        str, Any]:
        document_spans = [s for s in all_spans if s.document_id == target_span.document_id]

        preceding_spans = [
            s for s in document_spans
            if s.end <= target_span.start and target_span.start - s.end <= context_window
        ]

        following_spans = [
            s for s in document_spans
            if s.start >= target_span.end and s.start - target_span.end <= context_window
        ]

        return {
            'target_span': target_span,
            'preceding_context': sorted(preceding_spans, key=lambda s: s.start)[-3:],
            'following_context': sorted(following_spans, key=lambda s: s.start)[:3]
        }
