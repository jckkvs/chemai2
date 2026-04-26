// frontend_next/src/config/estimatorSchemas.ts
// Dynamic UI configuration for estimators and feature engines
// Schema-driven: adding a new estimator auto-generates its parameter UI

import type { EstimatorSchema, TaskType, UIComponentConfig } from '@/lib/types';

// Helper factories for common parameter types
const createNumberParam = (label: string, description: string, defaultVal: number, min?: number, max?: number, step = 1): UIComponentConfig => ({
  type: 'number',
  label,
  description,
  default: defaultVal,
  min,
  max,
  step,
});

const createSliderParam = (label: string, description: string, defaultVal: number, min: number, max: number, step = 0.01): UIComponentConfig => ({
  type: 'slider',
  label,
  description,
  default: defaultVal,
  min,
  max,
  step,
});

const createToggleParam = (label: string, description: string, defaultVal: boolean): UIComponentConfig => ({
  type: 'toggle',
  label,
  description,
  default: defaultVal,
});

const createSelectParam = (label: string, description: string, defaultVal: string, options: Array<{ value: string; label: string }>): UIComponentConfig => ({
  type: 'select',
  label,
  description,
  default: defaultVal,
  options,
});

const createMultiSelectParam = (label: string, description: string, defaultVal: string[], options: Array<{ value: string; label: string }>): UIComponentConfig => ({
  type: 'multi-select',
  label,
  description,
  default: defaultVal,
  options,
});

// Estimator schemas - add new estimators here for automatic UI generation
export const estimatorSchemas: Record<string, EstimatorSchema> = {
  // ── Regression Models ─────────────────────────────────
  LinearRegression: {
    name: 'Linear Regression',
    key: 'LinearRegression',
    task_types: ['regression'],
    description: 'Ordinary least squares Linear Regression',
    params: {
      fit_intercept: createToggleParam('Fit Intercept', 'Whether to calculate the intercept for this model', true),
      positive: createToggleParam('Positive Coefficients', 'Force coefficients to be positive', false),
    },
    constraints_supported: { monotonic: false, linear: true, group: false },
  },
  
  Ridge: {
    name: 'Ridge Regression',
    key: 'Ridge',
    task_types: ['regression'],
    description: 'Linear least squares with l2 regularization',
    params: {
      alpha: createSliderParam('Alpha (α)', 'Regularization strength; must be positive', 1.0, 0.0, 100.0, 0.1),
      fit_intercept: createToggleParam('Fit Intercept', 'Whether to calculate the intercept', true),
      solver: createSelectParam('Solver', 'Solver to use in optimization', 'auto', [
        { value: 'auto', label: 'Auto' },
        { value: 'svd', label: 'SVD' },
        { value: 'cholesky', label: 'Cholesky' },
        { value: 'lsqr', label: 'LSQR' },
        { value: 'saga', label: 'SAGA' },
      ]),
    },
    constraints_supported: { monotonic: false, linear: true, group: false },
  },
  
  Lasso: {
    name: 'Lasso Regression',
    key: 'Lasso',
    task_types: ['regression'],
    description: 'Linear Model trained with L1 prior as regularizer',
    params: {
      alpha: createSliderParam('Alpha (α)', 'Constant that multiplies the L1 penalty', 1.0, 0.0, 100.0, 0.1),
      fit_intercept: createToggleParam('Fit Intercept', 'Whether to calculate the intercept', true),
      max_iter: createNumberParam('Max Iterations', 'Maximum number of iterations', 1000, 100, 10000, 100),
      tol: createSliderParam('Tolerance', 'Precision of the solution', 1e-4, 1e-6, 1e-2, 1e-6),
    },
    constraints_supported: { monotonic: false, linear: true, group: false },
  },
  
  RandomForestRegressor: {
    name: 'Random Forest Regressor',
    key: 'RandomForestRegressor',
    task_types: ['regression'],
    description: 'A random forest regressor with monotonic constraint support',
    params: {
      n_estimators: createNumberParam('Number of Trees', 'Number of trees in the forest', 100, 10, 1000, 10),
      max_depth: createNumberParam('Max Depth', 'Maximum depth of the tree (null = unlimited)', null as any, 1, 50, 1),
      min_samples_split: createNumberParam('Min Samples Split', 'Minimum samples required to split', 2, 2, 100, 1),
      min_samples_leaf: createNumberParam('Min Samples Leaf', 'Minimum samples required at leaf', 1, 1, 50, 1),
      max_features: createSelectParam('Max Features', 'Number of features to consider at each split', 'sqrt', [
        { value: 'sqrt', label: '√n (sqrt)' },
        { value: 'log2', label: 'log₂(n)' },
        { value: 'auto', label: 'All features' },
      ]),
      bootstrap: createToggleParam('Bootstrap', 'Whether bootstrap samples are used', true),
      random_state: createNumberParam('Random State', 'Controls randomness for reproducibility', 42, 0, 9999, 1),
    },
    constraints_supported: { monotonic: true, linear: false, group: false },
  },
  
  GradientBoostingRegressor: {
    name: 'Gradient Boosting Regressor',
    key: 'GradientBoostingRegressor',
    task_types: ['regression'],
    description: 'Gradient Boosting for regression with monotonic constraints',
    params: {
      loss: createSelectParam('Loss Function', 'Loss function to be optimized', 'squared_error', [
        { value: 'squared_error', label: 'Squared Error' },
        { value: 'absolute_error', label: 'Absolute Error' },
        { value: 'huber', label: 'Huber' },
        { value: 'quantile', label: 'Quantile' },
      ]),
      learning_rate: createSliderParam('Learning Rate', 'Step size shrinkage', 0.1, 0.001, 1.0, 0.001),
      n_estimators: createNumberParam('Number of Trees', 'Number of boosting stages', 100, 10, 1000, 10),
      max_depth: createNumberParam('Max Depth', 'Maximum depth of trees', 3, 1, 20, 1),
      min_samples_split: createNumberParam('Min Samples Split', 'Minimum samples to split', 2, 2, 100, 1),
      subsample: createSliderParam('Subsample', 'Fraction of samples for each tree', 1.0, 0.1, 1.0, 0.05),
    },
    constraints_supported: { monotonic: true, linear: false, group: false },
  },
  
  XGBRegressor: {
    name: 'XGBoost Regressor',
    key: 'XGBRegressor',
    task_types: ['regression'],
    description: 'Extreme Gradient Boosting with advanced monotonic constraints',
    params: {
      n_estimators: createNumberParam('Number of Trees', 'Number of boosting rounds', 100, 10, 2000, 10),
      max_depth: createNumberParam('Max Depth', 'Maximum tree depth', 6, 1, 20, 1),
      learning_rate: createSliderParam('Learning Rate', 'Step size shrinkage (eta)', 0.3, 0.001, 1.0, 0.001),
      subsample: createSliderParam('Subsample', 'Subsample ratio of training instances', 1.0, 0.1, 1.0, 0.05),
      colsample_bytree: createSliderParam('Col Sample by Tree', 'Subsample ratio of columns', 1.0, 0.1, 1.0, 0.05),
      reg_alpha: createSliderParam('L1 Regularization (α)', 'L1 regularization term', 0.0, 0.0, 10.0, 0.1),
      reg_lambda: createSliderParam('L2 Regularization (λ)', 'L2 regularization term', 1.0, 0.0, 10.0, 0.1),
      random_state: createNumberParam('Random State', 'Random seed for reproducibility', 42, 0, 9999, 1),
    },
    constraints_supported: { monotonic: true, linear: false, group: true },
  },
  
  // ── Classification Models ─────────────────────────────────
  LogisticRegression: {
    name: 'Logistic Regression',
    key: 'LogisticRegression',
    task_types: ['classification'],
    description: 'Logistic Regression for binary and multiclass classification',
    params: {
      penalty: createSelectParam('Penalty', 'Norm used for penalization', 'l2', [
        { value: 'l1', label: 'L1 (Lasso)' },
        { value: 'l2', label: 'L2 (Ridge)' },
        { value: 'elasticnet', label: 'ElasticNet' },
        { value: 'none', label: 'None' },
      ]),
      C: createSliderParam('Inverse Regularization (C)', 'Smaller values = stronger regularization', 1.0, 0.01, 100.0, 0.01),
      solver: createSelectParam('Solver', 'Algorithm to use in optimization', 'lbfgs', [
        { value: 'lbfgs', label: 'L-BFGS' },
        { value: 'liblinear', label: 'LIBLINEAR' },
        { value: 'saga', label: 'SAGA' },
        { value: 'newton-cg', label: 'Newton-CG' },
      ]),
      max_iter: createNumberParam('Max Iterations', 'Maximum number of iterations', 100, 10, 1000, 10),
      class_weight: createSelectParam('Class Weight', 'Weights for classes', null as any, [
        { value: null, label: 'None' },
        { value: 'balanced', label: 'Balanced' },
      ]),
    },
    constraints_supported: { monotonic: false, linear: true, group: false },
  },
  
  RandomForestClassifier: {
    name: 'Random Forest Classifier',
    key: 'RandomForestClassifier',
    task_types: ['classification'],
    description: 'A random forest classifier with monotonic constraint support',
    params: {
      n_estimators: createNumberParam('Number of Trees', 'Number of trees in the forest', 100, 10, 1000, 10),
      max_depth: createNumberParam('Max Depth', 'Maximum depth of the tree', null as any, 1, 50, 1),
      min_samples_split: createNumberParam('Min Samples Split', 'Minimum samples to split', 2, 2, 100, 1),
      min_samples_leaf: createNumberParam('Min Samples Leaf', 'Minimum samples at leaf', 1, 1, 50, 1),
      class_weight: createSelectParam('Class Weight', 'Weights for classes', null as any, [
        { value: null, label: 'None' },
        { value: 'balanced', label: 'Balanced' },
        { value: 'balanced_subsample', label: 'Balanced Subsample' },
      ]),
      bootstrap: createToggleParam('Bootstrap', 'Use bootstrap samples', true),
      random_state: createNumberParam('Random State', 'Random seed', 42, 0, 9999, 1),
    },
    constraints_supported: { monotonic: true, linear: false, group: false },
  },
  
  XGBClassifier: {
    name: 'XGBoost Classifier',
    key: 'XGBClassifier',
    task_types: ['classification'],
    description: 'Extreme Gradient Boosting for classification',
    params: {
      n_estimators: createNumberParam('Number of Trees', 'Number of boosting rounds', 100, 10, 2000, 10),
      max_depth: createNumberParam('Max Depth', 'Maximum tree depth', 6, 1, 20, 1),
      learning_rate: createSliderParam('Learning Rate', 'Step size shrinkage', 0.3, 0.001, 1.0, 0.001),
      subsample: createSliderParam('Subsample', 'Subsample ratio of training instances', 1.0, 0.1, 1.0, 0.05),
      colsample_bytree: createSliderParam('Col Sample', 'Subsample ratio of columns', 1.0, 0.1, 1.0, 0.05),
      reg_alpha: createSliderParam('L1 Regularization', 'L1 penalty', 0.0, 0.0, 10.0, 0.1),
      reg_lambda: createSliderParam('L2 Regularization', 'L2 penalty', 1.0, 0.0, 10.0, 0.1),
      random_state: createNumberParam('Random State', 'Random seed', 42, 0, 9999, 1),
    },
    constraints_supported: { monotonic: true, linear: false, group: true },
  },
};

// Feature engine schemas (for SMILES/chemical descriptors)
export const featureEngineSchemas: Record<string, any> = {
  rdkit_descriptors: {
    name: 'RDKit Basic Descriptors',
    description: 'Physicochemical descriptors: MW, LogP, TPSA, H-bond donors/acceptors, etc.',
    params: {
      normalize: createToggleParam('Normalize', 'Scale features to [0,1] range', true),
      selected_descriptors: createMultiSelectParam('Select Descriptors', 'Choose which descriptors to compute', [], [
        { value: 'MolWt', label: 'Molecular Weight' },
        { value: 'LogP', label: 'LogP (octanol-water)' },
        { value: 'TPSA', label: 'Topological Polar Surface Area' },
        { value: 'NumHDonors', label: 'H-Bond Donors' },
        { value: 'NumHAcceptors', label: 'H-Bond Acceptors' },
        { value: 'NumRotatableBonds', label: 'Rotatable Bonds' },
        { value: 'RingCount', label: 'Ring Count' },
      ]),
    },
  },
  xtb_features: {
    name: 'GFN2-xTB Quantum Descriptors',
    description: 'Semi-empirical quantum mechanical descriptors: Energy, HOMO/LUMO, Gap, Dipole, etc.',
    params: {
      charge: createNumberParam('Charge', 'Total charge of the molecule', 0, -10, 10, 1),
      multiplicity: createNumberParam('Multiplicity', 'Spin multiplicity (1=singlet, 2=doublet, etc.)', 1, 1, 10, 1),
      optimize: createToggleParam('Optimize', 'Perform geometry optimization before descriptor calculation', true),
    },
  },
};

// Helper functions
export function getEstimatorSchema(key: string, taskType: TaskType): EstimatorSchema | undefined {
  const schema = estimatorSchemas[key];
  if (!schema) return undefined;
  if (!schema.task_types.includes(taskType)) return undefined;
  return schema;
}

export function getEstimatorsForTask(taskType: TaskType): EstimatorSchema[] {
  return Object.values(estimatorSchemas).filter(
    (schema) => schema.task_types.includes(taskType)
  );
}

export function getFeatureEngineSchema(key: string): any {
  return featureEngineSchemas[key] || null;
}

export function generateFormConfig(schema: EstimatorSchema, taskType: TaskType): Record<string, UIComponentConfig> {
  const filtered = { ...schema.params };
  return filtered;
}
