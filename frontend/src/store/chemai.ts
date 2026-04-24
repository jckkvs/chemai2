// frontend/src/store/chemai.ts
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
    pipelineConfig: {
      cv_folds: 5,
      num_scaler: 'standard',
      num_imputer: 'median',
      cat_encoder: 'onehot',
      feature_selector: 'none',
      selected_models: [],
      monotonic_constraints: {}
    },
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
        console.error('Session init failed', e)
      }
    },

    async uploadFile(file: File) {
      this.isLoading = true
      this.error = ''
      try {
        const formData = new FormData()
        formData.append('file', file)
        
        // session_id を Body に含める形式に変更
        formData.append('session_id', this.sessionId)
        
        const res = await axios.post(`${this.apiBase}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        
        const d = res.data
        this.filename = d.filename
        this.rows = d.rows
        this.cols = d.cols
        this.targetCol = d.target_col
        this.taskType = d.task_type
        this.preview = d.preview
        this.metrics = d.metrics
        this.columns = d.columns
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
          target_col: target,
          task_type: task || this.taskType
        })
        this.targetCol = target
        if (task) this.taskType = task as 'regression' | 'classification'
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Config update failed'
      }
    },

    async fetchPipelineConfig() {
      try {
        const res = await axios.get(`${this.apiBase}/pipeline/config`, {
          params: { session_id: this.sessionId }
        })
        this.pipelineConfig = res.data
      } catch (e: any) {
        console.error('Failed to fetch pipeline config', e)
      }
    },

    async updatePipelineConfig(cfg: any) {
      try {
        await axios.post(`${this.apiBase}/pipeline/config`, {
          session_id: this.sessionId,
          config: cfg
        })
        this.pipelineConfig = cfg
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Pipeline config update failed'
      }
    }
  }
})
