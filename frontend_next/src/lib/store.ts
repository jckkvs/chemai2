// src/lib/store.ts
import { create } from 'zustand'
import type { TaskType, PipelineConfig, AnalysisResult } from './types'

interface ChemAIState {
  // Session
  sessionId: string | null
  setSessionId: (id: string) => void
  
  // Data
  filename: string | null
  df: Record<string, any>[] | null
  columns: string[]
  targetCol: string | null
  taskType: TaskType
  metrics: {
    rows: number
    cols: number
    missing_rate: number
    numeric_cols: number
  } | null
  
  // Pipeline
  pipelineConfig: PipelineConfig
  analysisResult: AnalysisResult | null
  isLoading: boolean
  error: string | null
  
  // Actions
  setLoadedData: (data: {
    filename: string
    df: Record<string, any>[]
    columns: string[]
    targetCol: string
    taskType: TaskType
    metrics: { rows: number; cols: number; missing_rate: number; numeric_cols: number }
  }) => void
  updatePipelineConfig: (config: Partial<PipelineConfig>) => void
  setAnalysisResult: (result: AnalysisResult) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  clearData: () => void
}

const defaultPipelineConfig: PipelineConfig = {
  cv_folds: 5,
  num_scaler: 'standard',
  num_imputer: 'median',
  cat_encoder: 'onehot',
  feature_selector: 'none',
  selected_models: [],
  monotonic_constraints: {},
  do_polynomial: false,
  poly_degree: 2,
  do_eda: true,
  do_prep: true,
  do_eval: true,
}

export const useChemAIStore = create<ChemAIState>((set) => ({
  // Initial state
  sessionId: null,
  filename: null,
  df: null,
  columns: [],
  targetCol: null,
  taskType: 'regression',
  metrics: null,
  pipelineConfig: defaultPipelineConfig,
  analysisResult: null,
  isLoading: false,
  error: null,
  
  // Actions
  setSessionId: (id) => set({ sessionId: id }),
  
  setLoadedData: (data) => set({
    filename: data.filename,
    df: data.df,
    columns: data.columns,
    targetCol: data.targetCol,
    taskType: data.taskType,
    metrics: data.metrics,
    error: null,
  }),
  
  updatePipelineConfig: (config) => set((state) => ({
    pipelineConfig: { ...state.pipelineConfig, ...config },
  })),
  
  setAnalysisResult: (result) => set({ analysisResult: result }),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setError: (error) => set({ error }),
  
  clearData: () => set({
    filename: null,
    df: null,
    columns: [],
    targetCol: null,
    taskType: 'regression',
    metrics: null,
    analysisResult: null,
    error: null,
  }),
}))
