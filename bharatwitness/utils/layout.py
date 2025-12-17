# utils/layout.py
# BharatWitness layout analysis utilities

import regex as re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def overlap_ratio(self, other: 'BoundingBox') -> float:
        overlap_x = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        overlap_y = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        overlap_area = overlap_x * overlap_y

        union_area = self.area() + other.area() - overlap_area
        return overlap_area / union_area if union_area > 0 else 0


@dataclass
class LayoutElement:
    bbox: BoundingBox
    text: str
    element_type: str
    confidence: float
    reading_order: int


class LayoutAnalyzer:
    def __init__(self):
        self.column_threshold = 50
        self.line_height_threshold = 5
        self.heading_height_multiplier = 1.5

    def analyze_page_layout(self, ocr_results: List[Dict[str, Any]]) -> List[LayoutElement]:
        elements = []

        for i, result in enumerate(ocr_results):
            if len(result) < 2:
                continue

            bbox_coords, (text, confidence) = result
            bbox = self._create_bbox_from_coords(bbox_coords)

            element_type = self._classify_layout_element(text, bbox, ocr_results)

            element = LayoutElement(
                bbox=bbox,
                text=text,
                element_type=element_type,
                confidence=confidence,
                reading_order=i
            )
            elements.append(element)

        sorted_elements = self._sort_reading_order(elements)
        return sorted_elements

    def _create_bbox_from_coords(self, coords: List[List[float]]) -> BoundingBox:
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]

        return BoundingBox(
            x1=min(x_coords),
            y1=min(y_coords),
            x2=max(x_coords),
            y2=max(y_coords)
        )

    def _classify_layout_element(self, text: str, bbox: BoundingBox, all_results: List) -> str:
        text_stripped = text.strip()

        if not text_stripped:
            return 'empty'

        height = bbox.y2 - bbox.y1
        width = bbox.x2 - bbox.x1

        avg_height = self._calculate_average_height(all_results)

        if height > avg_height * self.heading_height_multiplier:
            return 'heading'

        if len(text_stripped) < 50 and text_stripped.isupper():
            return 'title'

        if re.match(r'^\d+\.', text_stripped):
            return 'numbered_item'

        if re.match(r'^\s*TABLE\s*\d*', text_stripped, re.IGNORECASE):
            return 'table_header'

        if self._is_likely_table_content(text_stripped):
            return 'table_cell'

        if width < 100 and len(text_stripped) < 20:
            return 'label'

        return 'paragraph'

    def _calculate_average_height(self, ocr_results: List) -> float:
        heights = []
        for result in ocr_results:
            if len(result) >= 2:
                bbox_coords = result[0]
                y_coords = [point[1] for point in bbox_coords]
                height = max(y_coords) - min(y_coords)
                heights.append(height)

        return np.mean(heights) if heights else 20.0

    def _is_likely_table_content(self, text: str) -> bool:
        tab_count = text.count('\t')
        space_sequences = len(re.findall(r'\s{2,}', text))

        return tab_count > 0 or space_sequences >= 2

    def _sort_reading_order(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        elements_with_position = []

        for element in elements:
            y_center = (element.bbox.y1 + element.bbox.y2) / 2
            x_center = (element.bbox.x1 + element.bbox.x2) / 2

            elements_with_position.append((element, y_center, x_center))

        # Sort by y-position first (top to bottom), then x-position (left to right)
        elements_with_position.sort(key=lambda item: (item[1], item[2]))

        # Extract just the LayoutElement from each tuple
        sorted_elements = [item[0] for item in elements_with_position]

        for i, element in enumerate(sorted_elements):
            element.reading_order = i

        return sorted_elements

    def detect_columns(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        if not elements:
            return []

        x_centers = [(element.bbox.x1 + element.bbox.x2) / 2 for element in elements]
        x_centers_sorted = sorted(set(x_centers))

        if len(x_centers_sorted) < 2:
            return [elements]

        columns = []
        current_column = []
        current_x = x_centers_sorted[0]

        for element in sorted(elements, key=lambda e: (e.bbox.x1 + e.bbox.x2) / 2):
            element_x = (element.bbox.x1 + element.bbox.x2) / 2

            if abs(element_x - current_x) > self.column_threshold:
                if current_column:
                    columns.append(current_column)
                current_column = [element]
                current_x = element_x
            else:
                current_column.append(element)

        if current_column:
            columns.append(current_column)

        return columns

    def extract_document_structure(self, elements: List[LayoutElement]) -> Dict[str, Any]:
        structure = {
            'title': None,
            'headings': [],
            'sections': [],
            'tables': [],
            'paragraphs': []
        }

        for element in elements:
            if element.element_type == 'title':
                if not structure['title']:
                    structure['title'] = element.text
            elif element.element_type == 'heading':
                structure['headings'].append({
                    'text': element.text,
                    'position': element.reading_order,
                    'bbox': element.bbox
                })
            elif element.element_type == 'numbered_item':
                structure['sections'].append({
                    'text': element.text,
                    'position': element.reading_order,
                    'bbox': element.bbox
                })
            elif element.element_type in ['table_header', 'table_cell']:
                structure['tables'].append({
                    'text': element.text,
                    'position': element.reading_order,
                    'bbox': element.bbox,
                    'type': element.element_type
                })
            elif element.element_type == 'paragraph':
                structure['paragraphs'].append({
                    'text': element.text,
                    'position': element.reading_order,
                    'bbox': element.bbox
                })

        return structure
