# evaluation/metrics.py
# BharatWitness comprehensive evaluation metrics suite

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import regex as re
from collections import defaultdict, Counter
import time
import yaml
import logging
from datetime import datetime
from sklearn.metrics import ndcg_score, precision_recall_fscore_support
from scipy import stats
import json

from pipeline.claim_verification import VerificationSummary, ClaimVerificationResult
from pipeline.answer_builder import BuiltAnswer
from pipeline.temporal_engine import TemporalSpan
from utils.span_utils import TextSpan


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class ComprehensiveMetrics:
    faithfulness: float
    contradiction_rate: float
    ndcg_at_10: float
    span_f1: float
    latency_p95: float
    temporal_accuracy: float
    code_mixed_robustness_drop: float
    calibration_ece: float
    additional_metrics: Dict[str, float]


class FaithfulnessMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.faithfulness")

    def compute(self, verification_results: List[ClaimVerificationResult]) -> float:
        if not verification_results:
            return 0.0

        supported_claims = sum(1 for result in verification_results if result.label == 'supported')
        total_claims = len(verification_results)

        attributable_claim_ratio = supported_claims / total_claims

        confidence_weighted_score = np.mean([
            result.calibrated_confidence if result.label == 'supported' else 0.0
            for result in verification_results
        ])

        faithfulness_score = 0.7 * attributable_claim_ratio + 0.3 * confidence_weighted_score

        return min(1.0, max(0.0, faithfulness_score))


class ContradictionMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.contradiction")

    def compute(self, verification_results: List[ClaimVerificationResult]) -> float:
        if not verification_results:
            return 0.0

        refuted_claims = sum(1 for result in verification_results if result.label == 'refuted')
        total_claims = len(verification_results)

        contradiction_rate = refuted_claims / total_claims

        return contradiction_rate


class NDCGMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.ndcg")

    def compute(self, retrieval_results: List[Any], ground_truth_relevance: List[int], k: int = 10) -> float:
        if not retrieval_results or not ground_truth_relevance:
            return 0.0

        try:
            predicted_scores = [result.score for result in retrieval_results[:k]]
            true_relevance = ground_truth_relevance[:len(predicted_scores)]

            if len(predicted_scores) != len(true_relevance):
                min_len = min(len(predicted_scores), len(true_relevance))
                predicted_scores = predicted_scores[:min_len]
                true_relevance = true_relevance[:min_len]

            if len(set(true_relevance)) == 1:
                return 1.0

            ndcg = ndcg_score([true_relevance], [predicted_scores], k=k)
            return ndcg

        except Exception as e:
            self.logger.warning(f"NDCG calculation failed: {e}")
            return 0.0


class SpanF1Metric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.span_f1")

    def compute(self, predicted_spans: List[Tuple[int, int]], true_spans: List[Tuple[int, int]]) -> Dict[str, float]:
        if not predicted_spans and not true_spans:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0}

        if not predicted_spans:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        if not true_spans:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        pred_set = set(predicted_spans)
        true_set = set(true_spans)

        true_positives = len(pred_set.intersection(true_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (
                                                                                       true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"f1": f1, "precision": precision, "recall": recall}


class LatencyMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.latency")

    def compute_p95(self, latencies: List[float]) -> float:
        if not latencies:
            return 0.0

        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95) - 1
        p95_index = max(0, min(p95_index, len(sorted_latencies) - 1))

        return sorted_latencies[p95_index]

    def compute_statistics(self, latencies: List[float]) -> Dict[str, float]:
        if not latencies:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "std": 0.0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "mean": np.mean(sorted_latencies),
            "median": np.median(sorted_latencies),
            "p95": sorted_latencies[int(n * 0.95) - 1] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99) - 1] if n > 0 else 0.0,
            "std": np.std(sorted_latencies)
        }


class TemporalAccuracyMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.temporal")

    def compute(self, temporal_spans: List[TemporalSpan], as_of_date: datetime) -> float:
        if not temporal_spans or not as_of_date:
            return 1.0

        valid_spans = 0
        total_spans = len(temporal_spans)

        for span in temporal_spans:
            if self._is_temporally_valid(span, as_of_date):
                valid_spans += 1

        return valid_spans / total_spans if total_spans > 0 else 1.0

    def _is_temporally_valid(self, span: TemporalSpan, as_of_date: datetime) -> bool:
        if span.effective_date and as_of_date < span.effective_date:
            return False

        if span.expiry_date and as_of_date > span.expiry_date:
            return False

        if span.repeal_date and as_of_date > span.repeal_date:
            return False

        return True


class RobustnessMetric:
    def __init__(self):
        self.logger = logging.getLogger("bharatwitness.robustness")

    def compute_code_mixed_drop(self, clean_performance: Dict[str, float],
                                code_mixed_performance: Dict[str, float]) -> float:
        clean_f1 = clean_performance.get("f1", 1.0)
        code_mixed_f1 = code_mixed_performance.get("f1", 0.0)

        robustness_drop = max(0.0, clean_f1 - code_mixed_f1)

        return robustness_drop

    def create_code_mixed_query(self, query: str) -> str:
        english_hindi_mapping = {
            "bank": "बैंक", "rule": "नियम", "regulation": "विनियम",
            "money": "पैसा", "account": "खाता", "document": "दस्तावेज",
            "requirement": "आवश्यकता", "government": "सरकार"
        }

        words = query.split()
        code_mixed_words = []

        for i, word in enumerate(words):
            if i % 3 == 0 and word.lower() in english_hindi_mapping:
                code_mixed_words.append(english_hindi_mapping[word.lower()])
            else:
                code_mixed_words.append(word)

        return " ".join(code_mixed_words)


class CalibrationMetric:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.logger = logging.getLogger("bharatwitness.calibration")

    def compute_ece(self, confidences: List[float], correctness: List[bool]) -> float:
        if len(confidences) != len(correctness) or not confidences:
            return 0.0

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in confidences]
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean([correctness[i] for i, in_b in enumerate(in_bin) if in_b])
                avg_confidence_in_bin = np.mean([confidences[i] for i, in_b in enumerate(in_bin) if in_b])

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class MetricsSuite:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.evaluation_config = self.config.get("evaluation", {})
        self.output_dir = self.evaluation_config.get("metrics_output", "data/processed/reports")

        self.faithfulness_metric = FaithfulnessMetric()
        self.contradiction_metric = ContradictionMetric()
        self.ndcg_metric = NDCGMetric()
        self.span_f1_metric = SpanF1Metric()
        self.latency_metric = LatencyMetric()
        self.temporal_metric = TemporalAccuracyMetric()
        self.robustness_metric = RobustnessMetric()
        self.calibration_metric = CalibrationMetric()

        self.logger = logging.getLogger("bharatwitness.metrics")

    def evaluate_comprehensive(self, evaluation_data: Dict[str, Any]) -> ComprehensiveMetrics:
        verification_results = evaluation_data.get("verification_results", [])
        retrieval_results = evaluation_data.get("retrieval_results", [])
        ground_truth_relevance = evaluation_data.get("ground_truth_relevance", [])
        predicted_spans = evaluation_data.get("predicted_spans", [])
        true_spans = evaluation_data.get("true_spans", [])
        latencies = evaluation_data.get("latencies", [])
        temporal_spans = evaluation_data.get("temporal_spans", [])
        as_of_date = evaluation_data.get("as_of_date")
        clean_performance = evaluation_data.get("clean_performance", {"f1": 1.0})
        code_mixed_performance = evaluation_data.get("code_mixed_performance", {"f1": 0.9})
        confidences = evaluation_data.get("confidences", [])
        correctness = evaluation_data.get("correctness", [])

        faithfulness = self.faithfulness_metric.compute(verification_results)
        contradiction_rate = self.contradiction_metric.compute(verification_results)
        ndcg_at_10 = self.ndcg_metric.compute(retrieval_results, ground_truth_relevance, k=10)

        span_f1_results = self.span_f1_metric.compute(predicted_spans, true_spans)
        span_f1 = span_f1_results["f1"]

        latency_p95 = self.latency_metric.compute_p95(latencies)

        temporal_accuracy = self.temporal_metric.compute(temporal_spans, as_of_date) if as_of_date else 1.0

        robustness_drop = self.robustness_metric.compute_code_mixed_drop(clean_performance, code_mixed_performance)

        calibration_ece = self.calibration_metric.compute_ece(confidences, correctness)

        additional_metrics = {
            "span_precision": span_f1_results["precision"],
            "span_recall": span_f1_results["recall"],
            "latency_mean": np.mean(latencies) if latencies else 0.0,
            "supported_claims_ratio": sum(1 for r in verification_results if r.label == 'supported') / max(
                len(verification_results), 1),
            "uncertain_claims_ratio": sum(1 for r in verification_results if r.label == 'uncertain') / max(
                len(verification_results), 1)
        }

        return ComprehensiveMetrics(
            faithfulness=faithfulness,
            contradiction_rate=contradiction_rate,
            ndcg_at_10=ndcg_at_10,
            span_f1=span_f1,
            latency_p95=latency_p95,
            temporal_accuracy=temporal_accuracy,
            code_mixed_robustness_drop=robustness_drop,
            calibration_ece=calibration_ece,
            additional_metrics=additional_metrics
        )

    def save_metrics_report(self, metrics: ComprehensiveMetrics, output_path: str):
        report = {
            "timestamp": datetime.now().isoformat(),
            "core_metrics": {
                "faithfulness": metrics.faithfulness,
                "contradiction_rate": metrics.contradiction_rate,
                "ndcg_at_10": metrics.ndcg_at_10,
                "span_f1": metrics.span_f1,
                "latency_p95_seconds": metrics.latency_p95,
                "temporal_accuracy": metrics.temporal_accuracy,
                "code_mixed_robustness_drop": metrics.code_mixed_robustness_drop,
                "calibration_ece": metrics.calibration_ece
            },
            "additional_metrics": metrics.additional_metrics,
            "summary": {
                "meets_faithfulness_target": metrics.faithfulness >= 0.95,
                "meets_contradiction_target": metrics.contradiction_rate <= 0.02,
                "meets_ndcg_target": metrics.ndcg_at_10 >= 0.85,
                "meets_latency_target": metrics.latency_p95 <= 3.0,
                "meets_temporal_target": metrics.temporal_accuracy >= 0.98,
                "meets_robustness_target": metrics.code_mixed_robustness_drop <= 0.05
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics report saved to {output_path}")

    def create_metrics_dataframe(self, multiple_evaluations: List[ComprehensiveMetrics]) -> pd.DataFrame:
        rows = []

        for i, metrics in enumerate(multiple_evaluations):
            row = {
                "evaluation_id": i,
                "faithfulness": metrics.faithfulness,
                "contradiction_rate": metrics.contradiction_rate,
                "ndcg_at_10": metrics.ndcg_at_10,
                "span_f1": metrics.span_f1,
                "latency_p95": metrics.latency_p95,
                "temporal_accuracy": metrics.temporal_accuracy,
                "robustness_drop": metrics.code_mixed_robustness_drop,
                "calibration_ece": metrics.calibration_ece
            }

            row.update(metrics.additional_metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def benchmark_against_targets(self, metrics: ComprehensiveMetrics) -> Dict[str, bool]:
        targets = {
            "faithfulness_95": metrics.faithfulness >= 0.95,
            "contradiction_2pct": metrics.contradiction_rate <= 0.02,
            "ndcg_85": metrics.ndcg_at_10 >= 0.85,
            "latency_3sec": metrics.latency_p95 <= 3.0,
            "temporal_98": metrics.temporal_accuracy >= 0.98,
            "robustness_5pct": metrics.code_mixed_robustness_drop <= 0.05,
            "calibration_10pct": metrics.calibration_ece <= 0.10
        }

        return targets
