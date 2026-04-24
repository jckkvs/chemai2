import { defineStore } from 'pinia'
import axios from 'axios'

export const useChemaiStore = defineStore('chemai', {
  state: () => ({
    sessionId: localStorage.getItem('chemai_session_id') || '',
    filename: '',
    rows: 0,
    cols: 0,
    targetCol: '',
    taskType: 'regression' as 'regression' | 'classification',
    preview: [] as Record<string, any>[],
    metrics: {} as Record<string, number>,
    columns: [] as string[],
    isLoading: false,
    error: ''
  }),

  getters: {
    hasData: (state) => state.rows > 0,
    apiBase: () => 'http://localhost:8000/api'
  },

  actions: {
    async initSession() {
      try {
        const res = await axios.post(`${this.apiBase}/session/init`)
        this.sessionId = res.data.session_id
        localStorage.setItem('chemai_session_id', this.sessionId)
      } catch (e) {
        console.error("Session init failed", e)
      }
    },

    async uploadFile(file: File) {
      this.isLoading = true
      this.error = ''
      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('session_id', this.sessionId)
        
        const res = await axios.post(`${this.apiBase}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        
        const data = res.data
        this.filename = data.filename
        this.rows = data.rows
        this.cols = data.cols
        this.targetCol = data.target_col
        this.taskType = data.task_type
        this.preview = data.preview
        this.metrics = data.metrics
        // Get all columns from preview keys if not provided
        this.columns = Object.keys(data.preview[0] || {})
        
        this.isLoading = false
        return true
      } catch (e: any) {
        this.error = e.response?.data?.detail || e.message
        this.isLoading = false
        return false
      }
    },

    async updateConfig(target: string, task?: string) {
      try {
        await axios.post(`${this.apiBase}/config/columns`, {
          session_id: this.sessionId,
          config: { target_col: target, task_type: task || this.taskType }
        })
        this.targetCol = target
        if (task) this.taskType = task as 'regression' | 'classification'
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Config update failed'
      }
    },

    async runAnalysis(cfg: any) {
      this.isLoading = true
      try {
        const res = await axios.post(`${this.apiBase}/pipeline/run`, {
          session_id: this.sessionId,
          ...cfg
        })
        this.isLoading = false
        return res.data
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Analysis failed'
        this.isLoading = false
        return null
      }
    }
  }
})
