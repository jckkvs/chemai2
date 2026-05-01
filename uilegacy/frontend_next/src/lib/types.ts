// frontend_next/src/lib/types.ts

// ── Basic Types ─────────────────────────────────
export type TaskType = 'regression' | 'classification';
export type NumericType = 'continuous' | 'discrete' | 'binary' | 'count';
export type ColumnType = 'numeric' | 'categorical' | 'binary' | 'datetime' | 'text' | 'smiles';

export interface DataColumn {
  name: string;
  type: ColumnType;
  numericType?: NumericType;
  categories?: string[];
  missingCount: number;
  uniqueCount: number;
  sampleValues: any[];
}

// ── API Response Types ─────────────────────────────────
export interface UploadResponse {
  success: boolean;
  filename: string;
  rows: number;
  cols: number;
  target_col: string;
  task_type: TaskType;
  metrics: {
    rows: number;
    cols: number;
    missing_rate: number;
    numeric_cols: number;
  };
  preview: Record<string, any>[];
  columns: string[];
  column_details?: DataColumn[];
}

export interface DataInfo {
  filename: string;
  columns: string[];
  column_details: DataColumn[];
  target_col: string;
  task_type: TaskType;
  metrics: {
    rows: number;
    cols: number;
    missing_rate: number;
    numeric_cols: number;
  };
  preview: Record<string, any>[];
}

export interface ColumnConfig {
  target_col: string;
  task_type?: TaskType;
  exclude_cols?: string[];
  column_types?: Record<string, ColumnType>;
}

// ── Pipeline Configuration ─────────────────────────────────
export interface PreprocessingConfig {
  // Numeric columns
  num_scaler: 'standard' | 'robust' | 'minmax' | 'maxabs' | 'none';
  num_imputer: 'median' | 'mean' | 'knn' | 'iterative' | 'drop';
  num_transform: 'none' | 'boxcox' | 'yeojohnson' | 'quantile_uniform' | 'quantile_normal' | 'log1p';
  
  // Categorical columns
  cat_encoder: 'onehot' | 'ordinal' | 'target' | 'binary' | 'leave_one_out';
  cat_imputer: 'most_frequent' | 'constant' | 'drop';
  
  // Column-specific overrides
  column_overrides?: Record<string, {
    scaler?: string;
    imputer?: string;
    transform?: string;
    encoder?: string;
  }>;
}

export interface FeatureGenerationConfig {
  do_polynomial: boolean;
  poly_degree: number;
  poly_interaction_only: boolean;
  do_custom_interactions?: Array<[string, string]>;
}

export interface FeatureSelectionConfig {
  feature_selector: 'none' | 'variance' | 'selectkbest_f' | 'selectkbest_mi' | 
                    'select_from_model_lasso' | 'select_from_model_rf' | 'rfe' | 'boruta';
  n_features_to_select: number;
  selector_params?: Record<string, any>;
}

export interface MonotonicConstraint {
  feature: string;
  direction: -1 | 0 | 1; // -1: decreasing, 0: unknown monotonic, 1: increasing
  strength: 'hard' | 'soft';
  sigma_range: number;
  linear: boolean;
}

export interface PipelineConfig {
  // Cross-validation
  cv_strategy: 'kfold' | 'stratified' | 'group' | 'time_series' | 'loo' | 'lgo';
  cv_folds: number;
  cv_params?: Record<string, any>;
  
  // Preprocessing
  preprocessing: PreprocessingConfig;
  
  // Feature engineering
  feature_generation: FeatureGenerationConfig;
  feature_selection: FeatureSelectionConfig;
  
  // Model selection
  estimator: string;
  estimator_params: Record<string, any>;
  
  // Constraints
  monotonic_constraints: MonotonicConstraint[];
  
  // Execution flags
  do_eda: boolean;
  do_prep: boolean;
  do_eval: boolean;
  do_pca: boolean;
  do_shap: boolean;
}

// ── Analysis Results ─────────────────────────────────
export interface AnalysisResult {
  status: 'pending' | 'running' | 'completed' | 'failed';
  best_model?: string;
  score?: number;
  cv_scores?: number[];
  feature_importances?: Array<{ name: string; value: number }>;
  message: string;
  metadata?: {
    training_time?: number;
    prediction_time?: number;
    model_size?: number;
  };
}

export interface EDAResults {
  stats: Array<{
    column: string;
    count: number;
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    q25?: number;
    q50?: number;
    q75?: number;
  }>;
  correlation: {
    columns: string[];
    matrix: number[][];
  };
  dim_reduction: {
    pca: number[][];
    tsne: number[][];
    explained_variance: number[];
  };
}

// ── SMILES Feature Engine Types ─────────────────────────────────
export interface FeatureEngine {
  key: string;
  name: string;
  description: string;
  category: 'physicochemical' | 'structural' | 'electronic' | 'topological' | 'quantum';
  compute_cost: 'low' | 'medium' | 'high';
  recommended_for: string[];
  params?: Record<string, {
    type: 'number' | 'boolean' | 'string' | 'select' | 'multi-select';
    default: any;
    options?: any[];
    min?: number;
    max?: number;
    description: string;
  }>;
}

export interface FeatureEngineResult {
  feature_names: string[];
  feature_matrix: number[][];
  metadata: {
    computed: number;
    skipped: number;
    errors?: string[];
  };
}

// ── UI Component Types ─────────────────────────────────
export interface UIComponentConfig {
  type: 'input' | 'slider' | 'toggle' | 'select' | 'multi-select' | 'number' | 'textarea';
  label: string;
  description?: string;
  default: any;
  required?: boolean;
  disabled?: boolean;
  visible?: boolean;
  // Type-specific props
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: any; label: string }>;
  placeholder?: string;
}

export interface EstimatorSchema {
  name: string;
  key: string;
  task_types: TaskType[];
  description: string;
  params: Record<string, UIComponentConfig>;
  constraints_supported?: {
    monotonic: boolean;
    linear: boolean;
    group: boolean;
  };
}

// ── Store State Types ─────────────────────────────────
export interface ChemAIState {
  // Session
  sessionId: string | null;
  
  // Data
  filename: string | null;
  columns: DataColumn[];
  targetCol: string | null;
  taskType: TaskType;
  preview: Record<string, any>[];
  metrics: {
    rows: number;
    cols: number;
    missing_rate: number;
    numeric_cols: number;
  } | null;
  
  // Pipeline
  pipelineConfig: PipelineConfig;
  availableEstimators: EstimatorSchema[];
  availableFeatureEngines: FeatureEngine[];
  
  // Results
  analysisResult: AnalysisResult | null;
  edaResults: EDAResults | null;
  
  // UI State
  isLoading: boolean;
  error: string | null;
  activeTab: 'data' | 'eda' | 'pipeline' | 'results';
  
  // Actions
  setSessionId: (id: string) => void;
  setLoadedData: (data: {
    filename: string;
    columns: DataColumn[];
    targetCol: string;
    taskType: TaskType;
    preview: Record<string, any>[];
    metrics: any;
  }) => void;
  updatePipelineConfig: (updates: Partial<PipelineConfig>) => void;
  setAnalysisResult: (result: AnalysisResult) => void;
  setEDAResults: (results: EDAResults) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: ChemAIState['activeTab']) => void;
  clearData: () => void;
}
