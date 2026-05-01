// frontend_next/src/lib/api.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import type {
  UploadResponse,
  ColumnConfig,
  PipelineConfig,
  AnalysisResult,
  DataInfo,
  EDAResults,
  FeatureEngine,
  EstimatorSchema,
  TaskType,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export const api: AxiosInstance = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes for ML operations
});

// ── Request Interceptors ─────────────────────────────────
api.interceptors.request.use((config) => {
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
  
  if (sessionId) {
    // For GET requests, add session_id to query params
    if (config.method === 'get' && config.params) {
      config.params = { ...config.params, session_id: sessionId };
    }
    // For POST/PUT, add to request body if not already present
    else if (config.data && typeof config.data === 'object' && !('session_id' in config.data)) {
      config.data = { ...config.data, session_id: sessionId };
    }
  }
  
  // Add request ID for tracing
  config.headers['X-Request-ID'] = crypto.randomUUID?.() || Date.now().toString();
  
  return config;
});

// ── Response Interceptors ─────────────────────────────────
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    // Log error with request ID for debugging
    const requestId = error.config?.headers?.['X-Request-ID'] || 'unknown';
    console.error(`[API Error req:${requestId}]`, {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message: error.response?.data?.message || error.message,
    });
    
    // Transform error for consistent handling
    if (error.response?.data?.error) {
      return Promise.reject(new Error(error.response.data.message || error.response.data.error));
    }
    return Promise.reject(error);
  }
);

// ── Session Management ─────────────────────────────────
export async function initSession(): Promise<string> {
  const response = await api.post('/session/init');
  const sessionId = response.data.session_id;
  if (typeof window !== 'undefined') {
    localStorage.setItem('chemai_session_id', sessionId);
  }
  return sessionId;
}

export async function closeSession(sessionId?: string): Promise<void> {
  const id = sessionId || localStorage.getItem('chemai_session_id');
  if (!id) return;
  
  await api.delete(`/session/${id}`);
  if (typeof window !== 'undefined') {
    localStorage.removeItem('chemai_session_id');
  }
}

// ── Data Management ─────────────────────────────────
export async function uploadData(file: File, onProgress?: (percent: number) => void): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });
  
  return response.data as UploadResponse;
}

export async function getDataInfo(): Promise<DataInfo> {
  const response = await api.get('/data/info');
  return response.data as DataInfo;
}

export async function updateColumns(config: ColumnConfig): Promise<{ status: string; target_col: string; task_type: string }> {
  const response = await api.post('/config/columns', config);
  return response.data;
}

// ── Pipeline Configuration ─────────────────────────────────
export async function getPipelineConfig(): Promise<PipelineConfig> {
  const response = await api.get('/pipeline/config');
  return response.data as PipelineConfig;
}

export async function updatePipelineConfig(config: PipelineConfig): Promise<{ status: string; config: PipelineConfig }> {
  const response = await api.post('/pipeline/config', config);
  return response.data;
}

export async function runPipeline(config: PipelineConfig, onStatusUpdate?: (status: string) => void): Promise<AnalysisResult> {
  // For long-running operations, we could implement polling or WebSocket
  const response = await api.post('/pipeline/run', config);
  return response.data as AnalysisResult;
}

export async function getResults(sessionId?: string): Promise<AnalysisResult> {
  const response = await api.get('/results', { params: sessionId ? { session_id: sessionId } : {} });
  return response.data as AnalysisResult;
}

// ── EDA Functions ─────────────────────────────────
export async function getEDAStats(sessionId?: string): Promise<EDAResults['stats']> {
  const response = await api.get('/eda/stats', { params: sessionId ? { session_id: sessionId } : {} });
  return response.data;
}

export async function getEDACorrelation(sessionId?: string, method: 'pearson' | 'spearman' | 'kendall' = 'pearson'): Promise<EDAResults['correlation']> {
  const response = await api.get('/eda/correlation', { params: { method, ...(sessionId ? { session_id: sessionId } : {}) } });
  return response.data;
}

export async function getEDADimReduction(sessionId?: string): Promise<EDAResults['dim_reduction']> {
  const response = await api.get('/eda/dim_reduction', { params: sessionId ? { session_id: sessionId } : {} });
  return response.data;
}

// ── MLOps / Experiment Tracking ─────────────────────────────────
export async function listExperiments(sessionId: string): Promise<any[]> {
  const response = await api.get('/experiments', { params: { session_id: sessionId } });
  return response.data;
}

export async function createExperiment(sessionId: string, runData: any): Promise<any> {
  const response = await api.post('/experiments', runData, { params: { session_id: sessionId } });
  return response.data;
}

export async function getExperimentMetrics(sessionId: string, runId: string): Promise<any> {
  const response = await api.get(`/experiments/${runId}/metrics`, { params: { session_id: sessionId } });
  return response.data;
}

// ── Dynamic Feature Computation ─────────────────────────────────
export async function computeFeaturesOnDemand(engineKey: string, smilesList: string[], config?: any): Promise<any> {
  const response = await api.post('/features/compute', { 
    smiles_list: smilesList, 
    config 
  }, { params: { engine_key: engineKey } });
  return response.data;
}

// ── Parameter Schema Discovery (Dynamic UI) ─────────────────────────────────
export async function getAvailableModels(task: TaskType = 'regression'): Promise<EstimatorSchema[]> {
  const response = await api.get('/params/models', { params: { task } });
  return response.data as EstimatorSchema[];
}

export async function getModelSchema(modelKey: string, task: TaskType = 'regression'): Promise<EstimatorSchema> {
  const response = await api.get(`/params/models/${modelKey}/schema`, { params: { task } });
  return response.data as EstimatorSchema;
}

export async function getAvailableFeatureEngines(): Promise<FeatureEngine[]> {
  const response = await api.get('/params/adapters');
  return response.data as FeatureEngine[];
}

export async function getFeatureEngineSchema(engineKey: string): Promise<FeatureEngine> {
  const response = await api.get(`/params/adapters/${engineKey}/schema`);
  return response.data as FeatureEngine;
}

// ── Benchmark Data ─────────────────────────────────
export async function getBenchmarks(): Promise<Array<{ id: string; name: string; description: string; target: string }>> {
  const response = await api.get('/data/benchmarks');
  return response.data;
}

export async function loadBenchmark(datasetId: string): Promise<UploadResponse> {
  const response = await api.post('/data/benchmarks/load', null, { params: { dataset_id: datasetId } });
  return response.data as UploadResponse;
}

// ── Health Check ─────────────────────────────────
export async function healthCheck(): Promise<{ status: string; version: string; environment: string }> {
  const response = await api.get('/health');
  return response.data;
}
