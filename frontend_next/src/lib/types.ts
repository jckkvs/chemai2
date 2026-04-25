// src/lib/types.ts

export type TaskType = 'regression' | 'classification'

export interface UploadResponse {
  success: boolean
  filename: string
  rows: number
  cols: number
  target_col: string
  task_type: TaskType
  metrics: {
    rows: number
    cols: number
    missing_rate: number
    numeric_cols: number
  }
  preview: Record<string, any>[]
  columns: string[]
}

export interface ColumnConfig {
  target_col: string
  task_type?: TaskType
  exclude_cols?: string[]
}

export interface PipelineConfig {
  cv_folds: number
  num_scaler: 'standard' | 'robust' | 'minmax' | 'maxabs' | 'none'
  num_imputer: 'median' | 'mean' | 'knn' | 'iterative' | 'drop'
  cat_encoder: 'onehot' | 'ordinal' | 'target' | 'binary'
  feature_selector: 'none' | 'variance' | 'selectkbest_f' | 'selectkbest_mi' | 'select_from_model_lasso' | 'select_from_model_rf' | 'rfe' | 'boruta'
  selected_models: string[]
  monotonic_constraints: Record<string, -1 | 0 | 1>
  do_polynomial: boolean
  poly_degree: number
  do_eda: boolean
  do_prep: boolean
  do_eval: boolean
}

export interface AnalysisResult {
  status: 'pending' | 'running' | 'completed' | 'failed'
  best_model?: string
  score?: number
  cv_scores?: number[]
  feature_importances?: { name: string; value: number }[]
  message: string
}

export interface DataInfo {
  filename: string
  columns: string[]
  target_col: string
  task_type: TaskType
  metrics: {
    rows: number
    cols: number
    missing_rate: number
    numeric_cols: number
  }
  preview: Record<string, any>[]
}
