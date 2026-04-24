// frontend/src/api/client.ts
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
})

// Request interceptor for session ID
api.interceptors.request.use((config) => {
  const sessionId = localStorage.getItem('chemai_session_id')
  if (sessionId && config.params) {
    config.params = { ...config.params, session_id: sessionId }
  } else if (sessionId) {
    config.params = { session_id: sessionId }
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export async function initSession() {
  const res = await api.post('/session/init')
  localStorage.setItem('chemai_session_id', res.data.session_id)
  return res.data.session_id
}

export async function uploadData(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function getDataInfo() {
  const res = await api.get('/data/info')
  return res.data
}

export async function updateColumns(config: { target_col: string; task_type?: string }) {
  const res = await api.post('/config/columns', config)
  return res.data
}

export async function runPipeline(config: any) {
  const res = await api.post('/pipeline/run', config)
  return res.data
}

export async function getResults() {
  const res = await api.get('/results')
  return res.data
}
