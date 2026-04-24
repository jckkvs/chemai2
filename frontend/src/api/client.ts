// frontend/src/api/client.ts
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';
export const api = axios.create({ baseURL: API_BASE });

export async function initSession() {
  const res = await api.post('/session/init');
  localStorage.setItem('chemai_session', res.data.session_id);
  localStorage.setItem('chemai_session_id', res.data.session_id); // Sync with store
  return res.data.session_id;
}

export async function uploadData(file: File) {
  const session = localStorage.getItem('chemai_session') || await initSession();
  const formData = new FormData();
  formData.append('file', file);
  // main.py の upload_data(session_id: str = Query(...)) と同期
  const res = await api.post(`/upload`, formData, {
    params: { session_id: session },
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
}

export async function getDataInfo() {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session');
  const res = await api.get(`/data/info`, { params: { session_id: session } });
  return res.data;
}

export async function updateColumns(config: { target_col: string; task_type?: string }) {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session');
  const res = await api.post(`/config/columns`, { session_id: session, config });
  return res.data;
}

export async function runPipeline(config: any) {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session');
  const res = await api.post(`/pipeline/run`, { session_id: session, cfg: config });
  return res.data;
}

export async function getResults() {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session');
  const res = await api.get(`/results`, { params: { session_id: session } });
  return res.data;
}
