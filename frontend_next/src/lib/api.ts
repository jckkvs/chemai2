// frontend_next/src/lib/api.ts
import axios from 'axios';

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor for Session Management
api.interceptors.request.use((config) => {
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
  if (sessionId) {
    if (config.method === 'get') {
      config.params = { ...config.params, session_id: sessionId };
    } else {
      if (config.data && typeof config.data === 'object') {
        config.data = { ...config.data, session_id: sessionId };
      }
    }
  }
  return config;
});

export async function initSession(): Promise<string> {
  const response = await api.post('/api/session/init');
  const sessionId = response.data.session_id;
  if (typeof window !== 'undefined') {
    localStorage.setItem('chemai_session_id', sessionId);
  }
  return sessionId;
}

export async function uploadData(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
  if (sessionId) {
    formData.append('session_id', sessionId);
  }

  const response = await api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function getModels(task: 'regression' | 'classification' = 'regression') {
  const response = await api.get('/api/params/models', { params: { task } });
  return response.data;
}

export async function runPipeline(config: any) {
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
  const response = await api.post('/api/pipeline/run', {
    session_id: sessionId,
    ...config
  });
  return response.data;
}

export async function getResults() {
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null;
  const response = await api.get('/api/results', { params: { session_id: sessionId } });
  return response.data;
}
