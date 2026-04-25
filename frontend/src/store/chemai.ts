// frontend/src/store/chemai.ts
import { defineStore } from 'pinia'
import { initSession, uploadData, updateColumns, runPipeline, getResults, getPipelineConfig, updatePipelineConfig } from '../api/client'

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
    async initialize() {
      if (!this.sessionId) {
        this.sessionId = await initSession();
        localStorage.setItem('chemai_session_id', this.sessionId);
      }
    },

    async uploadFile(file: File) {
      this.isLoading = true
      this.error = ''
      try {
        const d = await uploadData(file);
        
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
        await updateColumns({ target_col: target, task_type: task || this.taskType })
        this.targetCol = target
        if (task) this.taskType = task as 'regression' | 'classification'
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Config update failed'
      }
    },

    async runAnalysis(cfg: any) {
      this.isLoading = true
      try {
        const res = await runPipeline(cfg);
        this.isLoading = false
        return res
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Analysis failed'
        this.isLoading = false
        return null
      }
    },

    async fetchPipelineConfig() {
      try {
        const res = await getPipelineConfig();
        this.pipelineConfig = res
      } catch (e: any) {
        console.error('Failed to fetch pipeline config', e)
      }
    },

    async updatePipelineConfig(cfg: any) {
      try {
        await updatePipelineConfig(cfg)
        this.pipelineConfig = cfg
      } catch (e: any) {
        this.error = e.response?.data?.detail || 'Pipeline config update failed'
      }
    }
  }
})
