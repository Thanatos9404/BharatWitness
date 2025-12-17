# ocr/ocr_pipeline.py
# BharatWitness OCR processing with layout preservation and trust gating

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import unicodedata
import regex as re
import io
from PIL import Image
import logging
import yaml
import pypdf
from paddleocr import PaddleOCR

class OCRPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ocr_config = self.config["ocr"]
        self.trust_threshold = self.ocr_config["trust_threshold"]
        self.languages = self.ocr_config["languages"]
        self.use_layout = self.ocr_config["use_layout"]

        # Use multilingual mode for Hindi+English support
        # PaddleOCR langs: 'en' for English, 'hi' for Hindi (Devanagari)
        ocr_lang = 'hi' if 'hi' in self.languages else 'en'
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=ocr_lang,
            use_gpu=False
        )

        self.logger = logging.getLogger("bharatwitness.ocr")

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _process_pdf(self, pdf_path: Path) -> Dict[str, Any]:

        document_data = {
            'file_path': str(pdf_path),
            'pages': [],
            'metadata': {}
        }

        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            document_data['metadata'] = {
                'num_pages': len(reader.pages),
                'title': reader.metadata.get('/Title', '') if reader.metadata else '',
                'author': reader.metadata.get('/Author', '') if reader.metadata else ''
            }

            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        page_data = self._process_text_content(text, page_num)
                    else:
                        page_data = self._process_scanned_page(pdf_path, page_num)

                    document_data['pages'].append(page_data)

                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {e}")
                    continue

        return document_data

    def _process_image(self, image_path: Path) -> Dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        ocr_results = self.ocr_engine.ocr(image)

        return {
            'file_path': str(image_path),
            'pages': [self._extract_text_with_layout(ocr_results[0], 0)],
            'metadata': {'num_pages': 1}
        }

    def _process_scanned_page(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")

            image = Image.open(io.BytesIO(img_data))
            image_array = np.array(image)

            ocr_results = self.ocr_engine.ocr(image_array)
            return self._extract_text_with_layout(ocr_results[0], page_num)

        except ImportError:
            self.logger.warning("PyMuPDF not available, using basic PDF text extraction")
            return {'page_num': page_num, 'sections': [], 'raw_text': '', 'trust_score': 0.0}

    def _process_text_content(self, text: str, page_num: int) -> Dict[str, Any]:
        normalized_text = self._normalize_unicode(text)
        dehyphenated_text = self._dehyphenate(normalized_text)
        sections = self._segment_by_layout(dehyphenated_text)

        return {
            'page_num': page_num,
            'sections': sections,
            'raw_text': dehyphenated_text,
            'trust_score': 1.0
        }

    def _extract_text_with_layout(self, ocr_result: List, page_num: int) -> Dict[str, Any]:
        if not ocr_result:
            return {'page_num': page_num, 'sections': [], 'raw_text': '', 'trust_score': 0.0}

        sections = []
        full_text = []
        total_confidence = 0.0
        valid_detections = 0

        for detection in ocr_result:
            bbox, (text, confidence) = detection

            if confidence >= self.trust_threshold:
                normalized_text = self._normalize_unicode(text)

                section = {
                    'text': normalized_text,
                    'bbox': bbox,
                    'confidence': confidence,
                    'byte_offset': len(' '.join(full_text).encode('utf-8'))
                }

                sections.append(section)
                full_text.append(normalized_text)
                total_confidence += confidence
                valid_detections += 1

        dehyphenated_text = self._dehyphenate(' '.join(full_text))
        trust_score = total_confidence / valid_detections if valid_detections > 0 else 0.0

        return {
            'page_num': page_num,
            'sections': sections,
            'raw_text': dehyphenated_text,
            'trust_score': trust_score
        }

    def _normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\u200b|\u200c|\u200d|\ufeff', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _dehyphenate(self, text: str) -> str:
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        return text

    def _segment_by_layout(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        paragraphs = text.split('\n\n')
        byte_offset = 0

        for para in paragraphs:
            if para.strip():
                section = {
                    'text': para.strip(),
                    'type': self._classify_section_type(para.strip()),
                    'byte_offset': byte_offset
                }
                sections.append(section)
                byte_offset += len(para.encode('utf-8')) + 2

        return sections

    def _classify_section_type(self, text: str) -> str:
        if re.match(r'^\d+\.', text.strip()):
            return 'numbered_section'
        elif text.isupper() and len(text) < 100:
            return 'heading'
        elif re.match(r'^[A-Z][^.!?]*[.!?]', text):
            return 'paragraph'
        else:
            return 'text'

