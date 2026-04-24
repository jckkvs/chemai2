import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';
export const api = axios.create({ baseURL: API_BASE });

export async function initSession() {
  const res = await api.post('/session/init');
  const sessionId = res.data.session_id;
  localStorage.setItem('chemai_session', sessionId);
  return sessionId;
}

export async function uploadData(file: File) {
  let session = localStorage.getItem('chemai_session');
  if (!session) {
    session = await initSession();
  }
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post(`/session/${session}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
}

export async function runPipeline(config: any) {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session initialized');
  const res = await api.post(`/session/${session}/pipeline/run`, config);
  return res.data;
}

export async function getResults() {
  const session = localStorage.getItem('chemai_session');
  if (!session) throw new Error('No session initialized');
  const res = await api.get(`/session/${session}/results`);
  return res.data;
}
