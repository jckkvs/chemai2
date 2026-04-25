// src/lib/api.ts
import axios from 'axios'
import type { UploadResponse, ColumnConfig, PipelineConfig, AnalysisResult, DataInfo } from './types'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api'

export const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

// Axios インターセプター: セッション管理
api.interceptors.request.use((config) => {
  const sessionId = typeof window !== 'undefined' ? localStorage.getItem('chemai_session_id') : null
  if (sessionId && config.params) {
    config.params = { ...config.params, session_id: sessionId }
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// ── API 関数 ──

export async function initSession(): Promise<string> {
  const res = await api.post('/session/init')
  if (typeof window !== 'undefined') {
    localStorage.setItem('chemai_session_id', res.data.session_id)
  }
  return res.data.session_id
}

export async function uploadData(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data as UploadResponse
}

export async function getDataInfo(): Promise<DataInfo> {
  const res = await api.get('/data/info')
  return res.data as DataInfo
}

export async function updateColumns(config: ColumnConfig): Promise<{ status: string; target_col: string; task_type: string }> {
  const res = await api.post('/config/columns', config)
  return res.data
}

export async function getPipelineConfig(): Promise<PipelineConfig> {
  const res = await api.get('/pipeline/config')
  return res.data as PipelineConfig
}

export async function updatePipelineConfig(config: PipelineConfig): Promise<{ status: string; config: PipelineConfig }> {
  const res = await api.post('/pipeline/config', config)
  return res.data
}

export async function runPipeline(config: PipelineConfig): Promise<AnalysisResult> {
  const res = await api.post('/pipeline/run', config)
  return res.data as AnalysisResult
}

export async function getResults(): Promise<AnalysisResult> {
  const res = await api.get('/results')
  return res.data as AnalysisResult
}
