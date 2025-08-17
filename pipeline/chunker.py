# pipeline/chunker.py
# BharatWitness text chunking with layout and metadata preservation

from typing import List, Dict, Any, Optional, Tuple
import yaml
import logging
from dataclasses import dataclass
from bharatwitness.pipeline.segment import DocumentSection
from bharatwitness.utils.span_utils import SpanManager
from bharatwitness.utils.layout import LayoutAnalyzer


@dataclass
class TextChunk:
    text: str
    chunk_id: str
    byte_start: int
    byte_end: int
    page_nums: List[int]
    section_types: List[str]
    level: int
    metadata: Dict[str, Any]
    spans: List[Dict[str, Any]]


class DocumentChunker:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        retrieval_config = self.config["retrieval"]
        self.chunk_size = retrieval_config["chunk_size"]
        self.chunk_overlap = retrieval_config["chunk_overlap"]

        self.span_manager = SpanManager()
        self.logger = logging.getLogger("bharatwitness.chunker")

    def chunk_document(self, sections: List[DocumentSection], document_id: str) -> List[TextChunk]:
        chunks = []

        hierarchical_chunks = self._create_hierarchical_chunks(sections, document_id)
        chunks.extend(hierarchical_chunks)

        sliding_chunks = self._create_sliding_window_chunks(sections, document_id)
        chunks.extend(sliding_chunks)

        return self._deduplicate_chunks(chunks)

    def _create_hierarchical_chunks(self, sections: List[DocumentSection], document_id: str) -> List[TextChunk]:
        chunks = []
        chunk_buffer = []
        current_level = -1
        chunk_counter = 0

        for section in sections:
            if section.level <= current_level and chunk_buffer:
                chunk = self._create_chunk_from_buffer(chunk_buffer, document_id, chunk_counter, 'hierarchical')
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1
                chunk_buffer = []

            chunk_buffer.append(section)
            current_level = section.level

            buffer_text = ' '.join([s.text for s in chunk_buffer])
            if len(buffer_text.encode('utf-8')) > self.chunk_size:
                chunk = self._create_chunk_from_buffer(chunk_buffer, document_id, chunk_counter, 'hierarchical')
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1
                chunk_buffer = [section]

        if chunk_buffer:
            chunk = self._create_chunk_from_buffer(chunk_buffer, document_id, chunk_counter, 'hierarchical')
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_sliding_window_chunks(self, sections: List[DocumentSection], document_id: str) -> List[TextChunk]:
        chunks = []
        chunk_counter = 0

        all_text = ' '.join([section.text for section in sections])
        text_bytes = all_text.encode('utf-8')

        start = 0
        while start < len(text_bytes):
            end = min(start + self.chunk_size, len(text_bytes))

            chunk_text = text_bytes[start:end].decode('utf-8', errors='ignore')

            if start + self.chunk_size < len(text_bytes):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size // 2:
                    chunk_text = chunk_text[:last_space]
                    end = start + len(chunk_text.encode('utf-8'))

            relevant_sections = self._find_sections_in_range(sections, start, end)

            if chunk_text.strip() and relevant_sections:
                chunk = TextChunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{document_id}_sliding_{chunk_counter}",
                    byte_start=start,
                    byte_end=end,
                    page_nums=list(set([s.page_num for s in relevant_sections])),
                    section_types=list(set([s.section_type for s in relevant_sections])),
                    level=min([s.level for s in relevant_sections]),
                    metadata={
                        'chunk_type': 'sliding',
                        'original_sections': len(relevant_sections),
                        'languages': list(set([s.metadata.get('language', 'unknown') for s in relevant_sections]))
                    },
                    spans=[self._create_span_from_section(s) for s in relevant_sections]
                )

                chunks.append(chunk)
                chunk_counter += 1

            start = max(start + self.chunk_size - self.chunk_overlap, start + 1)

        return chunks

    def _create_chunk_from_buffer(self, buffer: List[DocumentSection], document_id: str, chunk_counter: int,
                                  chunk_type: str) -> Optional[TextChunk]:
        if not buffer:
            return None

        combined_text = ' '.join([section.text for section in buffer])
        if not combined_text.strip():
            return None

        return TextChunk(
            text=combined_text.strip(),
            chunk_id=f"{document_id}_{chunk_type}_{chunk_counter}",
            byte_start=min([s.byte_start for s in buffer]),
            byte_end=max([s.byte_end for s in buffer]),
            page_nums=list(set([s.page_num for s in buffer])),
            section_types=list(set([s.section_type for s in buffer])),
            level=min([s.level for s in buffer]),
            metadata={
                'chunk_type': chunk_type,
                'original_sections': len(buffer),
                'languages': list(set([s.metadata.get('language', 'unknown') for s in buffer])),
                'parent_sections': [s.metadata.get('parent') for s in buffer if s.metadata.get('parent')]
            },
            spans=[self._create_span_from_section(s) for s in buffer]
        )

    def _find_sections_in_range(self, sections: List[DocumentSection], start: int, end: int) -> List[DocumentSection]:
        relevant_sections = []

        for section in sections:
            if (section.byte_start >= start and section.byte_start < end) or \
                    (section.byte_end > start and section.byte_end <= end) or \
                    (section.byte_start < start and section.byte_end > end):
                relevant_sections.append(section)

        return relevant_sections

    def _create_span_from_section(self, section: DocumentSection) -> Dict[str, Any]:
        return {
            'text': section.text,
            'byte_start': section.byte_start,
            'byte_end': section.byte_end,
            'page_num': section.page_num,
            'section_type': section.section_type,
            'confidence': section.confidence
        }

    def _deduplicate_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        seen_texts = set()
        unique_chunks = []

        for chunk in chunks:
            text_hash = hash(chunk.text)
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_chunks.append(chunk)

        self.logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks

    def get_chunk_context(self, chunk: TextChunk, all_chunks: List[TextChunk]) -> Dict[str, Any]:
        context = {
            'preceding_chunks': [],
            'following_chunks': [],
            'same_page_chunks': [],
            'same_section_chunks': []
        }

        for other_chunk in all_chunks:
            if other_chunk.chunk_id == chunk.chunk_id:
                continue

            if other_chunk.byte_end <= chunk.byte_start:
                context['preceding_chunks'].append(other_chunk.chunk_id)
            elif other_chunk.byte_start >= chunk.byte_end:
                context['following_chunks'].append(other_chunk.chunk_id)

            if set(other_chunk.page_nums) & set(chunk.page_nums):
                context['same_page_chunks'].append(other_chunk.chunk_id)

            if set(other_chunk.section_types) & set(chunk.section_types):
                context['same_section_chunks'].append(other_chunk.chunk_id)

        return context
