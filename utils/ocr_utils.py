# utils/ocr_utils.py
# BharatWitness OCR utilities for confidence gating and text processing

import regex as re
from typing import List, Dict, Tuple, Optional
import unicodedata
from pathlib import Path


def apply_trust_gate(sections: List[Dict], threshold: float) -> List[Dict]:
    trusted_sections = []
    for section in sections:
        if section.get('confidence', 1.0) >= threshold:
            trusted_sections.append(section)
    return trusted_sections


def detect_language(text: str) -> str:
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))

    total_chars = devanagari_chars + latin_chars
    if total_chars == 0:
        return 'unknown'

    if devanagari_chars / total_chars > 0.3:
        return 'hi'
    else:
        return 'en'


def extract_dates(text: str) -> List[Dict[str, str]]:
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
        r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            dates.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })

    return dates


def clean_ocr_noise(text: str) -> str:
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\'\"\/\\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    return text.strip()


def calculate_text_quality_score(text: str) -> float:
    if not text:
        return 0.0

    total_chars = len(text)
    alphabetic_chars = sum(1 for c in text if c.isalpha())
    digit_chars = sum(1 for c in text if c.isdigit())
    space_chars = sum(1 for c in text if c.isspace())
    punctuation_chars = sum(1 for c in text if c in '.,;:!?()-[]\'"/\\')

    if total_chars == 0:
        return 0.0

    alphabetic_ratio = alphabetic_chars / total_chars
    readable_ratio = (alphabetic_chars + digit_chars + space_chars + punctuation_chars) / total_chars

    quality_score = (alphabetic_ratio * 0.7 + readable_ratio * 0.3)
    return min(1.0, quality_score)


def extract_section_hierarchy(text: str) -> List[Dict[str, any]]:
    hierarchy = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        level = 0
        section_type = 'paragraph'

        if re.match(r'^\d+\.', line):
            level = 1
            section_type = 'numbered'
        elif re.match(r'^[A-Z\s]+$', line) and len(line) < 100:
            level = 0
            section_type = 'heading'
        elif re.match(r'^\([a-z]\)', line):
            level = 2
            section_type = 'subsection'

        hierarchy.append({
            'text': line,
            'line_number': i,
            'level': level,
            'type': section_type
        })

    return hierarchy


def merge_fragmented_words(sections: List[Dict]) -> List[Dict]:
    merged_sections = []

    for i, section in enumerate(sections):
        text = section['text']

        if i > 0 and text and text[0].islower():
            prev_section = merged_sections[-1]
            if prev_section['text'].endswith('-'):
                prev_section['text'] = prev_section['text'][:-1] + text
                continue

        merged_sections.append(section.copy())

    return merged_sections
