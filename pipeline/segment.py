# pipeline/segment.py
# BharatWitness document segmentation with hierarchy preservation

import regex as re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass
from bharatwitness.utils.span_utils import SpanManager
from bharatwitness.utils.layout import LayoutAnalyzer


@dataclass
class DocumentSection:
    text: str
    section_type: str
    level: int
    byte_start: int
    byte_end: int
    page_num: int
    confidence: float
    metadata: Dict[str, Any]


class DocumentSegmenter:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.layout_analyzer = LayoutAnalyzer()
        self.span_manager = SpanManager()
        self.logger = logging.getLogger("bharatwitness.segment")

        self.section_patterns = {
            'chapter': r'^(?:CHAPTER|अध्याय)\s*[IVXLC\d]+',
            'section': r'^\d+\.\s+[A-Z]',
            'subsection': r'^\d+\.\d+\.\s',
            'clause': r'^\([a-z]\)\s',
            'definition': r'^"[^"]+"\s+means',
            'proviso': r'^\s*Provided\s+that',
            'explanation': r'^\s*Explanation\s*[:-]',
            'table': r'^\s*TABLE\s*\d*',
            'schedule': r'^SCHEDULE\s*[IVXLC\d]*'
        }

    def segment_document(self, document_data: Dict[str, Any]) -> List[DocumentSection]:
        all_sections = []

        for page in document_data['pages']:
            page_sections = self._segment_page(page)
            all_sections.extend(page_sections)

        hierarchical_sections = self._build_hierarchy(all_sections)
        return hierarchical_sections

    def _segment_page(self, page_data: Dict[str, Any]) -> List[DocumentSection]:
        sections = []
        page_num = page_data['page_num']
        raw_text = page_data['raw_text']

        if 'sections' in page_data and page_data['sections']:
            for section in page_data['sections']:
                doc_section = self._create_section_from_ocr(section, page_num, page_data['trust_score'])
                if doc_section:
                    sections.append(doc_section)
        else:
            doc_sections = self._segment_raw_text(raw_text, page_num, page_data.get('trust_score', 1.0))
            sections.extend(doc_sections)

        return sections

    def _create_section_from_ocr(self, section: Dict[str, Any], page_num: int, trust_score: float) -> Optional[
        DocumentSection]:
        text = section['text'].strip()
        if not text:
            return None

        section_type, level = self._classify_section(text)

        return DocumentSection(
            text=text,
            section_type=section_type,
            level=level,
            byte_start=section.get('byte_offset', 0),
            byte_end=section.get('byte_offset', 0) + len(text.encode('utf-8')),
            page_num=page_num,
            confidence=section.get('confidence', trust_score),
            metadata={
                'bbox': section.get('bbox'),
                'language': self._detect_primary_language(text)
            }
        )

    def _segment_raw_text(self, text: str, page_num: int, trust_score: float) -> List[DocumentSection]:
        sections = []
        paragraphs = text.split('\n\n')
        byte_offset = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            section_type, level = self._classify_section(para)

            section = DocumentSection(
                text=para,
                section_type=section_type,
                level=level,
                byte_start=byte_offset,
                byte_end=byte_offset + len(para.encode('utf-8')),
                page_num=page_num,
                confidence=trust_score,
                metadata={'language': self._detect_primary_language(para)}
            )

            sections.append(section)
            byte_offset += len(para.encode('utf-8')) + 2

        return sections

    def _classify_section(self, text: str) -> Tuple[str, int]:
        text_upper = text.upper()
        text_stripped = text.strip()

        for pattern_type, pattern in self.section_patterns.items():
            if re.match(pattern, text_stripped, re.IGNORECASE):
                level = self._get_section_level(pattern_type)
                return pattern_type, level

        if len(text_stripped) < 200 and text_stripped.isupper():
            return 'heading', 0
        elif re.match(r'^\d+\.', text_stripped):
            return 'numbered_section', 1
        elif re.match(r'^\([a-z]\)', text_stripped):
            return 'clause', 2
        elif text_stripped.startswith('Note:') or text_stripped.startswith('NOTE:'):
            return 'note', 2
        else:
            return 'paragraph', 3

    def _get_section_level(self, section_type: str) -> int:
        level_mapping = {
            'chapter': 0,
            'section': 1,
            'subsection': 2,
            'clause': 3,
            'definition': 2,
            'proviso': 3,
            'explanation': 3,
            'table': 2,
            'schedule': 1
        }
        return level_mapping.get(section_type, 3)

    def _detect_primary_language(self, text: str) -> str:
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        total_chars = devanagari_chars + latin_chars
        if total_chars == 0:
            return 'unknown'

        return 'hi' if devanagari_chars / total_chars > 0.3 else 'en'

    def _build_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        hierarchical_sections = []
        current_parent = None

        for section in sections:
            if section.level == 0:
                current_parent = section
                section.metadata['parent'] = None
            elif current_parent and section.level > current_parent.level:
                section.metadata['parent'] = current_parent.text[:50]

            hierarchical_sections.append(section)

        return hierarchical_sections
