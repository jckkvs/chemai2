// frontend_next/src/lib/store.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChemAIState, PipelineConfig, AnalysisResult, EDAResults, DataColumn } from './types';

const defaultPipelineConfig: PipelineConfig = {
  cv_strategy: 'kfold',
  cv_folds: 5,
  preprocessing: {
    num_scaler: 'standard',
    num_imputer: 'median',
    num_transform: 'none',
    cat_encoder: 'onehot',
    cat_imputer: 'most_frequent',
  },
  feature_generation: {
    do_polynomial: false,
    poly_degree: 2,
    poly_interaction_only: true,
  },
  feature_selection: {
    feature_selector: 'none',
    n_features_to_select: 20,
  },
  estimator: 'RandomForestRegressor',
  estimator_params: {},
  monotonic_constraints: [],
  do_eda: true,
  do_prep: true,
  do_eval: true,
  do_pca: true,
  do_shap: true,
};

export const useChemAIStore = create<ChemAIState>()(
  persist(
    (set, get) => ({
      // Initial state
      sessionId: null,
      filename: null,
      columns: [],
      targetCol: null,
      taskType: 'regression',
      preview: [],
      metrics: null,
      pipelineConfig: defaultPipelineConfig,
      availableEstimators: [],
      availableFeatureEngines: [],
      analysisResult: null,
      edaResults: null,
      isLoading: false,
      error: null,
      activeTab: 'data',
      
      // Actions
      setSessionId: (sessionId) => set({ sessionId }),
      
      setLoadedData: ({ filename, columns, targetCol, taskType, preview, metrics }) => 
        set({
          filename,
          columns,
          targetCol,
          taskType,
          preview,
          metrics,
          error: null,
        }),
      
      updatePipelineConfig: (updates) => 
        set((state) => ({
          pipelineConfig: { ...state.pipelineConfig, ...updates },
        })),
      
      setAnalysisResult: (result) => set({ analysisResult: result }),
      
      setEDAResults: (results) => set({ edaResults: results }),
      
      setLoading: (isLoading) => set({ isLoading }),
      
      setError: (error) => set({ error }),
      
      setActiveTab: (activeTab) => set({ activeTab }),
      
      clearData: () => set({
        filename: null,
        columns: [],
        targetCol: null,
        preview: [],
        metrics: null,
        analysisResult: null,
        edaResults: null,
        error: null,
      }),
    }),
    {
      name: 'chemai-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist essential state, not large data
        sessionId: state.sessionId,
        pipelineConfig: state.pipelineConfig,
        activeTab: state.activeTab,
      }),
    }
  )
);
