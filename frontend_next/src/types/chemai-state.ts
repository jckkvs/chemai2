/**
 * NiceGUI の state 辞書構造を 1:1 マッピングした TypeScript 定義
 * FastAPI リクエスト/レスポンスと完全互換
 */
export interface ChemAIState {
  target_col: string;
  task_type: 'regression' | 'classification' | 'auto';
  smiles_col?: string;
  exclude_cols: string[];
  cv_folds: number;
  cv_method: string;
  num_scaler: 'standard' | 'robust' | 'minmax' | 'maxabs' | 'none';
  num_imputer: 'median' | 'mean' | 'knn' | 'iterative' | 'drop';
  cat_encoder: 'onehot' | 'ordinal' | 'target' | 'binary';
  cat_imputer: 'most_frequent' | 'constant' | 'drop';
  feature_selector: string;
  n_features_to_select: number;
  selected_models: string[];
  model_params: Record<string, Record<string, any>>;
  monotonic_constraints: Record<string, number>;
  do_eda: boolean;
  do_shap: boolean;
  do_pca: boolean;
  do_polynomial: boolean;
  poly_degree: number;
}
