"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Button } from "./ui/Button";
import { Textarea } from "./ui/Textarea";
import { Input } from "./ui/Input";
import {
  ArrowLeft,
  Send,
  Loader2,
  CheckCircle2,
  XCircle,
  Calendar,
  FileText,
  Clock,
} from "lucide-react";
import { askQuestion, AskRequest, AskResponse } from "@/lib/api";
import { format } from "date-fns";

interface QueryInterfaceProps {
  onBack: () => void;
}

export const QueryInterface = ({ onBack }: QueryInterfaceProps) => {
  const [query, setQuery] = useState("");
  const [asOfDate, setAsOfDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const request: AskRequest = {
        query: query.trim(),
        confidence_threshold: 0.5,
        max_results: 10,
      };

      if (asOfDate) {
        request.as_of_date = new Date(asOfDate).toISOString();
      }

      const result = await askQuestion(request);
      setResponse(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-10 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button variant="ghost" onClick={onBack} className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
          <h1 className="text-4xl font-bold mb-2">
            <span className="gradient-text">Query Interface</span>
          </h1>
          <p className="text-muted-foreground">
            Ask questions about Indian government policies and legal documents
          </p>
        </div>

        {/* Query Form */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-primary" />
              Enter Your Question
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Question
                </label>
                <Textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., What are the current KYC requirements for banks?"
                  rows={4}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  As-of Date (Optional)
                </label>
                <Input
                  type="date"
                  value={asOfDate}
                  onChange={(e) => setAsOfDate(e.target.value)}
                  className="w-full max-w-xs"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Query documents valid as of this date
                </p>
              </div>

              <Button
                type="submit"
                disabled={loading || !query.trim()}
                className="w-full sm:w-auto"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4 mr-2" />
                    Ask Question
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="mb-6 border-destructive/50 bg-destructive/10">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <XCircle className="w-5 h-5 text-destructive mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-destructive mb-1">
                      Error
                    </h3>
                    <p className="text-sm text-destructive/90">{error}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Response Display */}
        {response && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Answer Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  Answer
                </CardTitle>
              </CardHeader>
              <CardContent>
                {response.refusal_reason ? (
                  <div className="p-4 rounded-lg border border-yellow-500/50 bg-yellow-500/10">
                    <p className="text-yellow-600 dark:text-yellow-400">
                      <strong>Note:</strong> {response.refusal_reason}
                    </p>
                  </div>
                ) : (
                  <div className="prose prose-invert max-w-none">
                    <p className="text-foreground leading-relaxed">
                      {response.answer}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Clock className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Processing Time
                      </p>
                      <p className="text-xl font-bold">
                        {response.processing_time.toFixed(2)}s
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                      <CheckCircle2 className="w-5 h-5 text-green-500" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Supported Claims
                      </p>
                      <p className="text-xl font-bold">
                        {response.verification_summary.supported_claims || 0} /{" "}
                        {response.verification_summary.total_claims || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                      <FileText className="w-5 h-5 text-blue-500" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Evidence Spans
                      </p>
                      <p className="text-xl font-bold">
                        {response.temporal_metadata.final_spans || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Citations Card */}
            {response.citations && response.citations.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Citations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {response.citations.map((citation, index) => (
                      <div
                        key={index}
                        className="p-3 rounded-lg bg-muted/50 border border-border"
                      >
                        <p className="text-sm font-mono text-muted-foreground">
                          [{index + 1}] {citation}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Temporal Metadata */}
            {response.temporal_metadata && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calendar className="w-5 h-5 text-primary" />
                    Temporal Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Total Spans
                      </p>
                      <p className="text-2xl font-bold">
                        {response.temporal_metadata.total_spans}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Suppressed
                      </p>
                      <p className="text-2xl font-bold">
                        {response.temporal_metadata.suppressed_spans}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Conflicted
                      </p>
                      <p className="text-2xl font-bold">
                        {response.temporal_metadata.conflicted_spans}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Final
                      </p>
                      <p className="text-2xl font-bold text-primary">
                        {response.temporal_metadata.final_spans}
                      </p>
                    </div>
                  </div>
                  {response.temporal_metadata.as_of_date && (
                    <div className="mt-4 pt-4 border-t border-border">
                      <p className="text-sm text-muted-foreground">
                        As of Date:{" "}
                        <span className="font-semibold text-foreground">
                          {format(
                            new Date(response.temporal_metadata.as_of_date),
                            "PPP"
                          )}
                        </span>
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
};
