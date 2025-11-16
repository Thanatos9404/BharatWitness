import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface AskRequest {
  query: string;
  as_of_date?: string;
  language_filter?: string[];
  section_type_filter?: string[];
  confidence_threshold?: number;
  max_results?: number;
}

export interface Citation {
  document: string;
  section: string;
  confidence: number;
}

export interface AskResponse {
  answer: string;
  citations: string[];
  verification_summary: {
    total_claims: number;
    supported_claims: number;
    contradicted_claims: number;
    refusal_recommended: boolean;
  };
  temporal_metadata: {
    as_of_date: string | null;
    total_spans: number;
    suppressed_spans: number;
    conflicted_spans: number;
    final_spans: number;
  };
  processing_time: number;
  refusal_reason: string | null;
}

export interface DiffRequest {
  query: string;
  old_date: string;
  new_date: string;
}

export interface DiffResponse {
  query: string;
  old_date: string;
  new_date: string;
  text_diff: string[];
  evidence_diff: {
    added: string[];
    removed: string[];
    common: string[];
  };
  summary: {
    old_processing_time: number;
    new_processing_time: number;
    hash_changed: boolean;
    old_span_count: number;
    new_span_count: number;
  };
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  system_info: {
    components_initialized: boolean;
    config_loaded: boolean;
    seed: string | number;
  };
  indices_status: Record<string, any>;
}

export const askQuestion = async (request: AskRequest): Promise<AskResponse> => {
  const response = await api.post<AskResponse>("/ask", request);
  return response.data;
};

export const getDiff = async (request: DiffRequest): Promise<DiffResponse> => {
  const response = await api.post<DiffResponse>("/diff", request);
  return response.data;
};

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>("/health");
  return response.data;
};

export default api;
