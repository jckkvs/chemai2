/**
 * ChemAI Nexus API Client - Type-safe HTTP + WebSocket client for FastAPI backend
 * Features: Automatic retries, progress streaming, error handling, TypeScript types
 */

import { EventEmitter } from 'events';

// Type definitions matching backend schemas
export interface ApiError {
  error: string;
  detail?: string;
  status?: number;
}

export interface TaskProgress {
  type: 'progress' | 'complete' | 'error';
  task_id: string;
  progress: number;
  message: string;
  timestamp: string;
  data?: Record<string, any>;
  result?: any;
  success?: boolean;
}

export interface ApiConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  retryCount?: number;
  retryDelay?: number;
}

export class ChemAIClient extends EventEmitter {
  private config: Required<ApiConfig>;
  private wsConnections: Map<string, WebSocket> = new Map();
  private progressCallbacks: Map<string, (progress: TaskProgress) => void> = new Map();

  constructor(config: ApiConfig) {
    super();
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:8000',
      apiKey: config.apiKey || '',
      timeout: config.timeout || 30000,
      retryCount: config.retryCount ?? 3,
      retryDelay: config.retryDelay ?? 1000,
    };
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    if (this.config.apiKey) {
      headers['X-API-Key'] = this.config.apiKey;
    }
    // Inject session ID if possible (fallback to localstorage if not in context)
    const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
    if (sessionId) {
        // Many endpoints expect session_id in query, but some might use headers
        // headers['X-Session-ID'] = sessionId;
    }
    return headers;
  }

  private async fetchWithRetry<T>(
    endpoint: string,
    options: RequestInit = {},
    retryCount = 0
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
      
      const response = await fetch(url, {
        ...options,
        headers: { ...this.getHeaders(), ...options.headers },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const error: ApiError = await response.json().catch(() => ({
          error: `HTTP ${response.status}`,
          detail: response.statusText,
        }));
        throw error;
      }
      
      return response.json() as Promise<T>;
      
    } catch (error: any) {
      if (retryCount < this.config.retryCount && 
          (error.name === 'AbortError' || error.status >= 500)) {
        await new Promise(resolve => setTimeout(resolve, this.config.retryDelay * (retryCount + 1)));
        return this.fetchWithRetry<T>(endpoint, options, retryCount + 1);
      }
      throw error;
    }
  }

  // ========== Data Management Endpoints ==========
  
  async uploadData(file: File, options?: { name?: string; format?: string }): Promise<{ task_id: string }> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.name) formData.append('name', options.name);
    if (options?.format) formData.append('format', options.format);
    
    return this.fetchWithRetry('/api/v1/data/upload', {
      method: 'POST',
      body: formData,
      // No JSON content type for multipart
    });
  }

  async getDataList(): Promise<{ datasets: Array<{ id: string; name: string; rows: number; columns: number }> }> {
    return this.fetchWithRetry('/api/v1/data/list');
  }

  async getDataInfo(dataId: string): Promise<{ 
    schema: Array<{ name: string; type: string; nullable: boolean }>;
    sample: Record<string, any>[];
    statistics: Record<string, any>;
  }> {
    return this.fetchWithRetry(`/api/v1/data/${dataId}/info`);
  }

  // ========== Chemical Descriptor Endpoints ==========
  
  async calculateDescriptors(
    dataId: string,
    smilesColumn: string,
    engines: string[],
    chargeConfig?: { charge: number; multiplicity: number; pH?: number }
  ): Promise<{ task_id: string }> {
    return this.fetchWithRetry('/api/v1/descriptors/calculate', {
      method: 'POST',
      body: JSON.stringify({ data_id: dataId, smiles_column: smilesColumn, engines, charge_config: chargeConfig }),
    });
  }

  async getAvailableEngines(): Promise<{ engines: Array<{ name: string; available: boolean; description: string }> }> {
    return this.fetchWithRetry('/api/v1/descriptors/engines');
  }

  // ========== ML Pipeline Endpoints ==========
  
  async runAutoML(
    dataId: string,
    targetColumn: string,
    taskType: 'regression' | 'classification',
    config?: {
      models?: string[];
      cvFolds?: number;
      optimize?: boolean;
      maxTimeMinutes?: number;
    }
  ): Promise<{ task_id: string }> {
    return this.fetchWithRetry('/api/v1/ml/automl', {
      method: 'POST',
      body: JSON.stringify({ 
        data_id: dataId, 
        target_column: targetColumn, 
        task_type: taskType,
        config 
      }),
    });
  }

  async getTaskResult(taskId: string): Promise<{ 
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    result?: any;
    error?: string;
  }> {
    return this.fetchWithRetry(`/api/v1/ml/tasks/${taskId}`);
  }

  // ========== Model Interpretation Endpoints ==========
  
  async generateSHAP(
    modelId: string,
    dataId: string,
    options?: { nSamples?: number; featureSubset?: string[] }
  ): Promise<{ task_id: string }> {
    return this.fetchWithRetry('/api/v1/interpret/shap', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, data_id: dataId, options }),
    });
  }

  // ========== WebSocket Progress Subscription ==========
  
  subscribeToProgress(taskId: string, onProgress: (progress: TaskProgress) => void): () => void {
    if (this.wsConnections.has(taskId)) {
      this.progressCallbacks.set(taskId, onProgress);
      return () => this.progressCallbacks.delete(taskId);
    }

    const baseUrl = this.config.baseUrl.endsWith('/') ? this.config.baseUrl.slice(0, -1) : this.config.baseUrl;
    const wsUrl = baseUrl.replace('http', 'ws') + `/api/v1/ws/progress/${taskId}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
      try {
        const progress: TaskProgress = JSON.parse(event.data);
        onProgress(progress);
        this.emit('progress', progress);
        
        if (progress.type === 'complete' || progress.type === 'error') {
          this.unsubscribeFromProgress(taskId);
        }
      } catch (e) {
        console.error('Failed to parse progress message:', e);
      }
    };
    
    ws.onerror = (error) => {
      console.error(`WebSocket error for task ${taskId}:`, error);
      this.emit('error', { taskId, error });
    };
    
    ws.onclose = () => {
      this.wsConnections.delete(taskId);
      this.progressCallbacks.delete(taskId);
    };
    
    this.wsConnections.set(taskId, ws);
    this.progressCallbacks.set(taskId, onProgress);
    
    // Return unsubscribe function
    return () => this.unsubscribeFromProgress(taskId);
  }

  private unsubscribeFromProgress(taskId: string): void {
    const ws = this.wsConnections.get(taskId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close(1000, 'Unsubscribed');
    }
    this.wsConnections.delete(taskId);
    this.progressCallbacks.delete(taskId);
  }

  // ========== Utility Methods ==========
  
  cancelTask(taskId: string): Promise<void> {
    return this.fetchWithRetry(`/api/v1/ml/tasks/${taskId}/cancel`, { method: 'POST' });
  }

  async exportResults(taskId: string, format: 'csv' | 'json' | 'pdf'): Promise<Blob> {
    const response = await fetch(
      `${this.config.baseUrl}/api/v1/export/${taskId}?format=${format}`,
      { headers: this.getHeaders() }
    );
    if (!response.ok) throw await response.json();
    return response.blob();
  }

  // Cleanup all connections
  disconnect(): void {
    for (const [taskId, ws] of this.wsConnections) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Client disconnecting');
      }
    }
    this.wsConnections.clear();
    this.progressCallbacks.clear();
  }
}

// Default client instance for application-wide use
export const chemaiClient = new ChemAIClient({
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  apiKey: process.env.NEXT_PUBLIC_API_KEY,
});
