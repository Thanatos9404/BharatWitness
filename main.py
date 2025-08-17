# main.py
# BharatWitness FastAPI server with Q&A, diff, and health endpoints

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import time
import yaml
import logging
from pathlib import Path
import uvicorn

from pipeline.retrieval import HybridRetriever, QueryContext
from pipeline.claim_verification import ClaimVerificationPipeline
from pipeline.answer_builder import AnswerBuilder
from pipeline.temporal_engine import TemporalEngine
from pipeline.index_build import HybridIndexBuilder
from utils.logging_utils import setup_logging
from utils.seed_utils import set_deterministic_seed


class AskRequest(BaseModel):
    query: str = Field(..., description="Question to ask the system")
    as_of_date: Optional[datetime] = Field(None, description="Date for temporal filtering")
    language_filter: Optional[List[str]] = Field(default_factory= ["en", "hi"], description="Languages to filter")
    section_type_filter: Optional[List[str]] = Field(None, description="Document section types to filter")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold")
    max_results: Optional[int] = Field(10, description="Maximum number of results")


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    verification_summary: Dict[str, Any]
    temporal_metadata: Dict[str, Any]
    processing_time: float
    refusal_reason: Optional[str]


class DiffRequest(BaseModel):
    query: str = Field(..., description="Question for versioned comparison")
    old_date: datetime = Field(..., description="Earlier date for comparison")
    new_date: datetime = Field(..., description="Later date for comparison")


class DiffResponse(BaseModel):
    query: str
    old_date: datetime
    new_date: datetime
    text_diff: List[str]
    evidence_diff: Dict[str, List[str]]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    system_info: Dict[str, Any]
    indices_status: Dict[str, Any]


class BharatWitnessAPI:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        set_deterministic_seed(self.config["system"]["seed"])
        self.logger = setup_logging(config_path)

        self.retriever = None
        self.verifier = None
        self.answer_builder = None
        self.temporal_engine = None
        self.index_builder = None

        self._initialize_components()

    def _initialize_components(self):
        try:
            self.logger.info("Initializing BharatWitness components")

            self.index_builder = HybridIndexBuilder(self.config_path)
            if not self.index_builder.load_indices():
                self.logger.warning("Could not load pre-built indices")

            self.retriever = HybridRetriever(self.config_path)
            self.verifier = ClaimVerificationPipeline(self.config_path)
            self.answer_builder = AnswerBuilder(self.config_path)
            self.temporal_engine = TemporalEngine(self.config_path)

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def ask_question(self, request: AskRequest) -> AskResponse:
        start_time = time.time()

        try:
            query_context = QueryContext(
                query=request.query,
                as_of_date=request.as_of_date,
                language_filter=request.language_filter,
                section_type_filter=request.section_type_filter,
                confidence_threshold=request.confidence_threshold,
                max_results=request.max_results
            )

            retrieval_results = self.retriever.retrieve(query_context)

            if not retrieval_results:
                return AskResponse(
                    answer="No relevant information found for your query.",
                    citations=[],
                    verification_summary={"total_claims": 0, "refusal_recommended": True},
                    temporal_metadata={},
                    processing_time=time.time() - start_time,
                    refusal_reason="No relevant documents found"
                )

            temporal_spans = self.temporal_engine.create_temporal_spans(retrieval_results)

            if request.as_of_date:
                valid_spans, suppressed_spans = self.temporal_engine.filter_spans_by_date(
                    temporal_spans, request.as_of_date
                )
                resolved_spans, conflicted_spans = self.temporal_engine.resolve_conflicts(
                    valid_spans, request.as_of_date
                )
            else:
                resolved_spans = temporal_spans
                suppressed_spans = []
                conflicted_spans = []

            answer_text = " ".join([span.text for span in resolved_spans[:3]])
            verification_summary = self.verifier.verify_answer(answer_text, [])

            final_answer = self.answer_builder.build_answer(
                query=request.query,
                verified_spans=resolved_spans,
                verification_summary=verification_summary,
                verification_results=[]
            )

            temporal_metadata = {
                "as_of_date": request.as_of_date.isoformat() if request.as_of_date else None,
                "total_spans": len(temporal_spans),
                "suppressed_spans": len(suppressed_spans),
                "conflicted_spans": len(conflicted_spans),
                "final_spans": len(resolved_spans)
            }

            processing_time = time.time() - start_time

            return AskResponse(
                answer=final_answer.answer_text,
                citations=final_answer.citations,
                verification_summary=final_answer.verification_summary,
                temporal_metadata=temporal_metadata,
                processing_time=processing_time,
                refusal_reason=final_answer.refusal_reason
            )

        except Exception as e:
            self.logger.error(f"Error processing query '{request.query}': {e}")

            return AskResponse(
                answer="An error occurred while processing your query.",
                citations=[],
                verification_summary={"error": str(e)},
                temporal_metadata={},
                processing_time=time.time() - start_time,
                refusal_reason=f"System error: {str(e)}"
            )

    async def compute_diff(self, request: DiffRequest) -> DiffResponse:
        try:
            old_request = AskRequest(
                query=request.query,
                as_of_date=request.old_date,
                confidence_threshold=0.5,
                max_results=10
            )

            new_request = AskRequest(
                query=request.query,
                as_of_date=request.new_date,
                confidence_threshold=0.5,
                max_results=10
            )

            old_response = await self.ask_question(old_request)
            new_response = await self.ask_question(new_request)

            old_answer_obj = self.answer_builder.build_answer(
                query=request.query,
                verified_spans=[],
                verification_summary=self.verifier.verify_answer("", []),
                verification_results=[]
            )
            old_answer_obj.answer_text = old_response.answer

            new_answer_obj = self.answer_builder.build_answer(
                query=request.query,
                verified_spans=[],
                verification_summary=self.verifier.verify_answer("", []),
                verification_results=[]
            )
            new_answer_obj.answer_text = new_response.answer

            diff = self.answer_builder.create_versioned_diff(old_answer_obj, new_answer_obj)

            return DiffResponse(
                query=request.query,
                old_date=request.old_date,
                new_date=request.new_date,
                text_diff=diff["text_diff"],
                evidence_diff=diff["evidence_diff"],
                summary={
                    "old_processing_time": old_response.processing_time,
                    "new_processing_time": new_response.processing_time,
                    "hash_changed": diff["metadata_diff"]["hash_changed"],
                    "old_span_count": diff["metadata_diff"]["old_span_count"],
                    "new_span_count": diff["metadata_diff"]["new_span_count"]
                }
            )

        except Exception as e:
            self.logger.error(f"Error computing diff for query '{request.query}': {e}")
            raise HTTPException(status_code=500, detail=f"Diff computation failed: {str(e)}")

    def get_health_status(self) -> HealthResponse:
        try:
            indices_status = {}

            if self.index_builder:
                indices_status = self.index_builder.get_index_stats()

            if self.retriever:
                retrieval_stats = self.retriever.get_retrieval_stats()
                indices_status.update(retrieval_stats)

            system_info = {
                "components_initialized": all([
                    self.retriever is not None,
                    self.verifier is not None,
                    self.answer_builder is not None,
                    self.temporal_engine is not None
                ]),
                "config_loaded": self.config is not None,
                "seed": self.config.get("system", {}).get("seed", "unknown")
            }

            return HealthResponse(
                status="healthy" if system_info["components_initialized"] else "degraded",
                timestamp=datetime.now(),
                system_info=system_info,
                indices_status=indices_status
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                system_info={"error": str(e)},
                indices_status={}
            )


bharatwitness_api = BharatWitnessAPI()

app = FastAPI(
    title="BharatWitness",
    description="Production-grade RAG system for Indian government policy and legal documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return bharatwitness_api.get_health_status()


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    return await bharatwitness_api.ask_question(request)


@app.post("/diff", response_model=DiffResponse, tags=["Analysis"])
async def compute_versioned_diff(request: DiffRequest):
    return await bharatwitness_api.compute_diff(request)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "BharatWitness API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BharatWitness API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
