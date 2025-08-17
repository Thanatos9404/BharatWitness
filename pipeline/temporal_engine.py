# pipeline/temporal_engine.py
# BharatWitness temporal reasoning engine with precedence logic and versioned diffing

import regex as re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import yaml
import logging
import dateparser
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path

from bharatwitness.utils.temporal_utils import TemporalFilter, TemporalMetadata
from bharatwitness.utils.span_utils import TextSpan, SpanManager
from bharatwitness.pipeline.retrieval import RetrievalResult

@dataclass
class TemporalSpan:
    span_id: str
    text: str
    document_id: str
    effective_date: Optional[datetime]
    expiry_date: Optional[datetime]
    publication_date: Optional[datetime]
    repeal_date: Optional[datetime]
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    amendments: List[str] = field(default_factory=list)
    status: str = "active"
    precedence_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VersionedAnswer:
    answer_text: str
    as_of_date: datetime
    spans: List[TemporalSpan]
    suppressed_spans: List[TemporalSpan]
    conflicts_resolved: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnswerDiff:
    old_answer: VersionedAnswer
    new_answer: VersionedAnswer
    added_spans: List[TemporalSpan]
    removed_spans: List[TemporalSpan]
    modified_spans: List[Tuple[TemporalSpan, TemporalSpan]]
    diff_summary: Dict[str, Any]

class PrecedenceGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger("bharatwitness.precedence")

    def add_span(self, span: TemporalSpan):
        self.graph.add_node(span.span_id, span=span)

        for superseded in span.supersedes:
            if superseded in self.graph:
                self.graph.add_edge(span.span_id, superseded, relation='supersedes')

        if span.superseded_by and span.superseded_by in self.graph:
            self.graph.add_edge(span.superseded_by, span.span_id, relation='supersedes')

    def get_active_spans(self, as_of_date: datetime) -> List[TemporalSpan]:
        active_spans = []

        for node_id in self.graph.nodes():
            span = self.graph.nodes[node_id]['span']

            if self._is_span_active(span, as_of_date):
                if not self._has_active_superseding_span(span, as_of_date):
                    active_spans.append(span)

        return sorted(active_spans, key=lambda s: s.precedence_weight, reverse=True)

    def _is_span_active(self, span: TemporalSpan, as_of_date: datetime) -> bool:
        if span.effective_date and as_of_date < span.effective_date:
            return False
        if span.expiry_date and as_of_date > span.expiry_date:
            return False
        if span.repeal_date and as_of_date > span.repeal_date:
            return False
        return span.status in ['active', 'amended']

    def _has_active_superseding_span(self, span: TemporalSpan, as_of_date: datetime) -> bool:
        predecessors = list(self.graph.predecessors(span.span_id))

        for pred_id in predecessors:
            edge_data = self.graph.get_edge_data(pred_id, span.span_id)
            if edge_data and edge_data.get('relation') == 'supersedes':
                superseding_span = self.graph.nodes[pred_id]['span']
                if self._is_span_active(superseding_span, as_of_date):
                    return True

        return False

    def detect_conflicts(self, as_of_date: datetime) -> List[Tuple[TemporalSpan, TemporalSpan]]:
        conflicts = []
        active_spans = self.get_active_spans(as_of_date)

        for i, span1 in enumerate(active_spans):
            for span2 in active_spans[i+1:]:
                if self._spans_conflict(span1, span2):
                    conflicts.append((span1, span2))

        return conflicts

    def _spans_conflict(self, span1: TemporalSpan, span2: TemporalSpan) -> bool:
        if span1.document_id == span2.document_id:
            return False

        text1_clean = re.sub(r'\W+', ' ', span1.text.lower()).strip()
        text2_clean = re.sub(r'\W+', ' ', span2.text.lower()).strip()

        if self._calculate_text_similarity(text1_clean, text2_clean) > 0.8:
            return True

        return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

class TemporalEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.temporal_config = self.config["temporal"]
        self.enable_precedence = self.temporal_config["enable_precedence"]

        self.temporal_filter = TemporalFilter()
        self.span_manager = SpanManager()
        self.precedence_graph = PrecedenceGraph()

        self.logger = logging.getLogger("bharatwitness.temporal")

        self.precedence_weights = {
            'constitution': 10.0,
            'act': 8.0,
            'rule': 6.0,
            'regulation': 5.0,
            'notification': 4.0,
            'circular': 3.0,
            'guideline': 2.0,
            'order': 1.5,
            'advisory': 1.0
        }

    def create_temporal_spans(self, retrieval_results: List[RetrievalResult]) -> List[TemporalSpan]:
        temporal_spans = []

        for result in retrieval_results:
            for span_data in result.spans:
                temporal_metadata = self.temporal_filter.extract_temporal_metadata(
                    span_data['text'],
                    span_data
                )

                document_type = self._classify_document_type(result.text)
                precedence_weight = self.precedence_weights.get(document_type, 1.0)

                temporal_span = TemporalSpan(
                    span_id=f"{result.chunk_id}_{span_data.get('byte_start', 0)}",
                    text=span_data['text'],
                    document_id=result.chunk_id.split('_')[0],
                    effective_date=temporal_metadata.effective_date,
                    expiry_date=temporal_metadata.expiry_date,
                    publication_date=temporal_metadata.publication_date,
                    repeal_date=None,
                    supersedes=temporal_metadata.supersedes or [],
                    superseded_by=temporal_metadata.superseded_by,
                    status=temporal_metadata.status,
                    precedence_weight=precedence_weight,
                    metadata={
                        'document_type': document_type,
                        'confidence': result.confidence,
                        'retrieval_score': result.score,
                        'original_span': span_data
                    }
                )

                temporal_spans.append(temporal_span)

        return temporal_spans

    def _classify_document_type(self, text: str) -> str:
        text_lower = text.lower()

        patterns = {
            'constitution': r'\bconstitution\b',
            'act': r'\bact\b|\bstatute\b',
            'rule': r'\brules?\b|\bregulations?\b',
            'notification': r'\bnotification\b|\bgazette\b',
            'circular': r'\bcircular\b|\bmemorandum\b',
            'guideline': r'\bguidelines?\b|\bdirectives?\b',
            'order': r'\border\b|\bdecision\b',
            'advisory': r'\badvisory\b|\bguidance\b'
        }

        for doc_type, pattern in patterns.items():
            if re.search(pattern, text_lower):
                return doc_type

        return 'guideline'

    def filter_spans_by_date(self, spans: List[TemporalSpan], as_of_date: datetime) -> Tuple[List[TemporalSpan], List[TemporalSpan]]:
        valid_spans = []
        suppressed_spans = []

        if self.enable_precedence:
            for span in spans:
                self.precedence_graph.add_span(span)

            valid_spans = self.precedence_graph.get_active_spans(as_of_date)

            all_span_ids = {span.span_id for span in spans}
            valid_span_ids = {span.span_id for span in valid_spans}
            suppressed_span_ids = all_span_ids - valid_span_ids

            suppressed_spans = [span for span in spans if span.span_id in suppressed_span_ids]
        else:
            for span in spans:
                if self._is_span_valid_simple(span, as_of_date):
                    valid_spans.append(span)
                else:
                    suppressed_spans.append(span)

        return valid_spans, suppressed_spans

    def _is_span_valid_simple(self, span: TemporalSpan, as_of_date: datetime) -> bool:
        if span.effective_date and as_of_date < span.effective_date:
            return False
        if span.expiry_date and as_of_date > span.expiry_date:
            return False
        if span.repeal_date and as_of_date > span.repeal_date:
            return False
        return span.status == 'active'

    def resolve_conflicts(self, spans: List[TemporalSpan], as_of_date: datetime) -> Tuple[List[TemporalSpan], List[TemporalSpan]]:
        if not self.enable_precedence:
            return spans, []

        conflicts = self.precedence_graph.detect_conflicts(as_of_date)

        resolved_spans = []
        conflicted_spans = []
        conflict_pairs = set()

        for span1, span2 in conflicts:
            conflict_pairs.add(span1.span_id)
            conflict_pairs.add(span2.span_id)

            if span1.precedence_weight > span2.precedence_weight:
                winner, loser = span1, span2
            elif span2.precedence_weight > span1.precedence_weight:
                winner, loser = span2, span1
            else:
                if span1.effective_date and span2.effective_date:
                    winner = span1 if span1.effective_date > span2.effective_date else span2
                    loser = span2 if winner == span1 else span1
                else:
                    winner, loser = span1, span2

            if winner not in resolved_spans:
                resolved_spans.append(winner)
            if loser not in conflicted_spans:
                conflicted_spans.append(loser)

        for span in spans:
            if span.span_id not in conflict_pairs:
                resolved_spans.append(span)

        self.logger.info(f"Resolved {len(conflicts)} conflicts, suppressed {len(conflicted_spans)} spans")
        return resolved_spans, conflicted_spans

    def generate_versioned_answer(self, answer_text: str, spans: List[TemporalSpan], as_of_date: datetime) -> VersionedAnswer:
        valid_spans, suppressed_spans = self.filter_spans_by_date(spans, as_of_date)
        resolved_spans, conflicted_spans = self.resolve_conflicts(valid_spans, as_of_date)

        all_suppressed = suppressed_spans + conflicted_spans

        versioned_answer = VersionedAnswer(
            answer_text=self._annotate_answer_with_date(answer_text, as_of_date),
            as_of_date=as_of_date,
            spans=resolved_spans,
            suppressed_spans=all_suppressed,
            conflicts_resolved=len(conflicted_spans),
            metadata={
                'total_original_spans': len(spans),
                'temporally_filtered': len(suppressed_spans),
                'conflict_resolved': len(conflicted_spans),
                'final_spans': len(resolved_spans)
            }
        )

        return versioned_answer

    def _annotate_answer_with_date(self, answer_text: str, as_of_date: datetime) -> str:
        date_str = as_of_date.strftime("%B %d, %Y")
        return f"As of {date_str}: {answer_text}"

    def compute_answer_diff(self, old_answer: VersionedAnswer, new_answer: VersionedAnswer) -> AnswerDiff:
        old_span_texts = {span.text: span for span in old_answer.spans}
        new_span_texts = {span.text: span for span in new_answer.spans}

        old_texts = set(old_span_texts.keys())
        new_texts = set(new_span_texts.keys())

        added_texts = new_texts - old_texts
        removed_texts = old_texts - new_texts
        common_texts = old_texts & new_texts

        added_spans = [new_span_texts[text] for text in added_texts]
        removed_spans = [old_span_texts[text] for text in removed_texts]

        modified_spans = []
        for text in common_texts:
            old_span = old_span_texts[text]
            new_span = new_span_texts[text]

            if (old_span.effective_date != new_span.effective_date or
                old_span.status != new_span.status or
                old_span.precedence_weight != new_span.precedence_weight):
                modified_spans.append((old_span, new_span))

        diff_summary = {
            'spans_added': len(added_spans),
            'spans_removed': len(removed_spans),
            'spans_modified': len(modified_spans),
            'date_range': (old_answer.as_of_date, new_answer.as_of_date),
            'total_change_score': self._calculate_change_score(added_spans, removed_spans, modified_spans)
        }

        return AnswerDiff(
            old_answer=old_answer,
            new_answer=new_answer,
            added_spans=added_spans,
            removed_spans=removed_spans,
            modified_spans=modified_spans,
            diff_summary=diff_summary
        )

    def _calculate_change_score(self, added: List[TemporalSpan], removed: List[TemporalSpan], modified: List[Tuple]) -> float:
        add_score = sum(span.precedence_weight for span in added)
        remove_score = sum(span.precedence_weight for span in removed)
        modify_score = sum(new_span.precedence_weight for _, new_span in modified) * 0.5

        return add_score + remove_score + modify_score

    def create_diff_visualization(self, diff: AnswerDiff) -> Dict[str, Any]:
        visualization = {
            'timeline': {
                'old_date': diff.old_answer.as_of_date.isoformat(),
                'new_date': diff.new_answer.as_of_date.isoformat()
            },
            'changes': {
                'added': [
                    {
                        'text': span.text[:100] + "..." if len(span.text) > 100 else span.text,
                        'effective_date': span.effective_date.isoformat() if span.effective_date else None,
                        'document_type': span.metadata.get('document_type', 'unknown'),
                        'precedence_weight': span.precedence_weight
                    }
                    for span in diff.added_spans
                ],
                'removed': [
                    {
                        'text': span.text[:100] + "..." if len(span.text) > 100 else span.text,
                        'effective_date': span.effective_date.isoformat() if span.effective_date else None,
                        'document_type': span.metadata.get('document_type', 'unknown'),
                        'precedence_weight': span.precedence_weight
                    }
                    for span in diff.removed_spans
                ],
                'modified': [
                    {
                        'text': new_span.text[:100] + "..." if len(new_span.text) > 100 else new_span.text,
                        'old_status': old_span.status,
                        'new_status': new_span.status,
                        'old_effective': old_span.effective_date.isoformat() if old_span.effective_date else None,
                        'new_effective': new_span.effective_date.isoformat() if new_span.effective_date else None
                    }
                    for old_span, new_span in diff.modified_spans
                ]
            },
            'summary': diff.diff_summary
        }

        return visualization

    def get_temporal_statistics(self, spans: List[TemporalSpan]) -> Dict[str, Any]:
        stats = {
            'total_spans': len(spans),
            'document_types': defaultdict(int),
            'status_distribution': defaultdict(int),
            'date_ranges': {
                'earliest_effective': None,
                'latest_effective': None,
                'earliest_expiry': None,
                'latest_expiry': None
            },
            'precedence_distribution': defaultdict(int),
            'supersession_relationships': 0
        }

        effective_dates = []
        expiry_dates = []

        for span in spans:
            stats['document_types'][span.metadata.get('document_type', 'unknown')] += 1
            stats['status_distribution'][span.status] += 1

            if span.effective_date:
                effective_dates.append(span.effective_date)
            if span.expiry_date:
                expiry_dates.append(span.expiry_date)

            weight_range = f"{int(span.precedence_weight)}-{int(span.precedence_weight)+1}"
            stats['precedence_distribution'][weight_range] += 1

            if span.supersedes or span.superseded_by:
                stats['supersession_relationships'] += 1

        if effective_dates:
            stats['date_ranges']['earliest_effective'] = min(effective_dates).isoformat()
            stats['date_ranges']['latest_effective'] = max(effective_dates).isoformat()

        if expiry_dates:
            stats['date_ranges']['earliest_expiry'] = min(expiry_dates).isoformat()
            stats['date_ranges']['latest_expiry'] = max(expiry_dates).isoformat()

        return stats
