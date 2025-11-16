"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/Card";
import {
  Search,
  Clock,
  FileCheck,
  Globe,
  Shield,
  Zap,
  Database,
  TrendingUp,
} from "lucide-react";

const features = [
  {
    icon: Search,
    title: "Hybrid Dense + Sparse Retrieval",
    description:
      "SPLADE sparse expansion + multilingual-e5 dense embeddings with HNSW-FAISS indexing for superior accuracy.",
  },
  {
    icon: Clock,
    title: "Temporal Reasoning Engine",
    description:
      "Handles effective dates, repeals, supersessions, and cross-document conflict resolution with 'as-of' date queries.",
  },
  {
    icon: FileCheck,
    title: "Span-Level Provenance",
    description:
      "Every answer includes precise byte-offset citations with confidence scores for complete transparency.",
  },
  {
    icon: Shield,
    title: "NLI Claim Verification",
    description:
      "mDeBERTa-based verification to minimize hallucinations and ensure factual accuracy.",
  },
  {
    icon: Globe,
    title: "Multi-Lingual Support",
    description:
      "Robust Hindi-English code-mixed query processing with multi-script OCR capabilities.",
  },
  {
    icon: Zap,
    title: "Production-Ready Performance",
    description:
      "p95 latency ≤3.0 seconds end-to-end with comprehensive metrics and monitoring.",
  },
  {
    icon: Database,
    title: "Comprehensive Document Processing",
    description:
      "PaddleOCR with layout preservation, hierarchical chunking, and Unicode normalization.",
  },
  {
    icon: TrendingUp,
    title: "Versioned Answer Diffing",
    description:
      "Track how policies evolve over time with detailed change analysis and impact assessment.",
  },
];

export const Features = () => {
  return (
    <section className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Advanced Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Production-grade capabilities designed for governance-level quality
            and transparency.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card hover className="h-full">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-sm leading-relaxed">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
