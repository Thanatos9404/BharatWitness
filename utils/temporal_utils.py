# utils/temporal_utils.py
# BharatWitness temporal filtering and date precedence utilities

import regex as re
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
import dateparser
import logging
from dataclasses import dataclass


@dataclass
class TemporalMetadata:
    effective_date: Optional[datetime]
    expiry_date: Optional[datetime]
    publication_date: Optional[datetime]
    superseded_by: Optional[str]
    supersedes: Optional[List[str]]
    status: str


class TemporalFilter:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.temporal")

        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]

        self.status_indicators = {
            'active': ['in force', 'current', 'valid', 'effective'],
            'repealed': ['repealed', 'revoked', 'superseded', 'replaced'],
            'suspended': ['suspended', 'stayed', 'postponed'],
            'amended': ['amended', 'modified', 'updated']
        }

    def extract_temporal_metadata(self, text: str, metadata: Dict[str, Any]) -> TemporalMetadata:
        effective_date = self._extract_effective_date(text)
        expiry_date = self._extract_expiry_date(text)
        publication_date = self._extract_publication_date(text, metadata)

        superseded_by = self._extract_supersession_info(text)
        supersedes = self._extract_supersedes_info(text)

        status = self._determine_status(text)

        return TemporalMetadata(
            effective_date=effective_date,
            expiry_date=expiry_date,
            publication_date=publication_date,
            superseded_by=superseded_by,
            supersedes=supersedes,
            status=status
        )

    def _extract_effective_date(self, text: str) -> Optional[datetime]:
        patterns = [
            r'(?:effective|comes?\s+into\s+force|shall\s+come\s+into\s+force)(?:\s+on)?(?:\s+the)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?:effective|comes?\s+into\s+force|shall\s+come\s+into\s+force)(?:\s+on)?(?:\s+the)?\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'with\s+effect\s+from\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'w\.e\.f\.?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date

        return None

    def _extract_expiry_date(self, text: str) -> Optional[datetime]:
        patterns = [
            r'(?:expires?|ceases?\s+to\s+have\s+effect|valid\s+until)(?:\s+on)?(?:\s+the)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?:expires?|ceases?\s+to\s+have\s+effect|valid\s+until)(?:\s+on)?(?:\s+the)?\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'shall\s+remain\s+in\s+force\s+until\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date

        return None

    def _extract_publication_date(self, text: str, metadata: Dict[str, Any]) -> Optional[datetime]:
        if 'publication_date' in metadata:
            pub_date = metadata['publication_date']
            if isinstance(pub_date, str):
                return dateparser.parse(pub_date)
            elif isinstance(pub_date, datetime):
                return pub_date

        patterns = [
            r'published\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'gazette\s+(?:dated\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'notification\s+(?:dated\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date

        return None

    def _extract_supersession_info(self, text: str) -> Optional[str]:
        patterns = [
            r'superseded\s+by\s+([A-Z0-9\s,./()-]+?)(?:\s+dated|\s+of|\.|$)',
            r'replaced\s+by\s+([A-Z0-9\s,./()-]+?)(?:\s+dated|\s+of|\.|$)',
            r'substituted\s+by\s+([A-Z0-9\s,./()-]+?)(?:\s+dated|\s+of|\.|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_supersedes_info(self, text: str) -> Optional[List[str]]:
        patterns = [
            r'(?:supersedes?|replaces?|substitutes?)\s+([A-Z0-9\s,./()-]+?)(?:\s+dated|\s+of|\.|$)',
            r'in\s+supersession\s+of\s+([A-Z0-9\s,./()-]+?)(?:\s+dated|\s+of|\.|$)'
        ]

        supersedes_list = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                superseded_docs = match.group(1).strip()
                if ',' in superseded_docs:
                    supersedes_list.extend([doc.strip() for doc in superseded_docs.split(',')])
                else:
                    supersedes_list.append(superseded_docs)

        return supersedes_list if supersedes_list else None

    def _determine_status(self, text: str) -> str:
        text_lower = text.lower()

        for status, indicators in self.status_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return status

        return 'active'

    def is_valid_as_of(self, result, as_of_date: datetime) -> bool:
        try:
            temporal_metadata = self.extract_temporal_metadata(result.text, result.metadata)

            if temporal_metadata.expiry_date and as_of_date > temporal_metadata.expiry_date:
                return False

            if temporal_metadata.effective_date and as_of_date < temporal_metadata.effective_date:
                return False

            if temporal_metadata.status in ['repealed', 'suspended']:
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Temporal validation failed: {e}")
            return True

    def get_precedence_order(self, results: List[Any], as_of_date: datetime) -> List[Any]:
        results_with_metadata = []

        for result in results:
            temporal_metadata = self.extract_temporal_metadata(result.text, result.metadata)

            precedence_score = self._calculate_precedence_score(temporal_metadata, as_of_date)

            results_with_metadata.append((result, temporal_metadata, precedence_score))

        results_with_metadata.sort(key=lambda x: x[2], reverse=True)

        return [item for item in results_with_metadata]

    def _calculate_precedence_score(self, metadata: TemporalMetadata, as_of_date: datetime) -> float:
        score = 0.0

        if metadata.effective_date:
            days_since_effective = (as_of_date - metadata.effective_date).days
            if days_since_effective >= 0:
                score += 100.0
                score += min(days_since_effective / 365.0, 10.0)

        if metadata.status == 'active':
            score += 50.0
        elif metadata.status == 'amended':
            score += 30.0
        elif metadata.status in ['repealed', 'suspended']:
            score -= 100.0

        if metadata.superseded_by:
            score -= 75.0

        if metadata.supersedes:
            score += 25.0

        return score

    def detect_conflicts(self, results: List[Any]) -> List[Dict[str, Any]]:
        conflicts = []

        for i, result1 in enumerate(results):
            metadata1 = self.extract_temporal_metadata(result1.text, result1.metadata)

            for j, result2 in enumerate(results[i + 1:], i + 1):
                metadata2 = self.extract_temporal_metadata(result2.text, result2.metadata)

                if self._are_conflicting(metadata1, metadata2):
                    conflicts.append({
                        'result1_id': result1.chunk_id,
                        'result2_id': result2.chunk_id,
                        'conflict_type': self._get_conflict_type(metadata1, metadata2),
                        'resolution': self._suggest_resolution(metadata1, metadata2)
                    })

        return conflicts

    def _are_conflicting(self, metadata1: TemporalMetadata, metadata2: TemporalMetadata) -> bool:
        if metadata1.superseded_by and metadata2.supersedes:
            return metadata1.superseded_by in metadata2.supersedes

        if metadata2.superseded_by and metadata1.supersedes:
            return metadata2.superseded_by in metadata1.supersedes

        if (metadata1.effective_date and metadata2.effective_date and
                metadata1.effective_date == metadata2.effective_date and
                metadata1.status != metadata2.status):
            return True

        return False

    def _get_conflict_type(self, metadata1: TemporalMetadata, metadata2: TemporalMetadata) -> str:
        if metadata1.superseded_by or metadata2.superseded_by:
            return 'supersession'
        elif metadata1.status != metadata2.status:
            return 'status_conflict'
        else:
            return 'temporal_overlap'

    def _suggest_resolution(self, metadata1: TemporalMetadata, metadata2: TemporalMetadata) -> str:
        if metadata1.superseded_by:
            return 'prefer_metadata2'
        elif metadata2.superseded_by:
            return 'prefer_metadata1'
        elif metadata1.effective_date and metadata2.effective_date:
            if metadata1.effective_date > metadata2.effective_date:
                return 'prefer_metadata1'
            else:
                return 'prefer_metadata2'
        else:
            return 'manual_review_required'
