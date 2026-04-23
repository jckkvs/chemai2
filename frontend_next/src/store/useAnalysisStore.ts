import { create } from 'zustand';

interface AnalysisState {
  jobId: string | null;
  status: 'idle' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  result: any | null;
  error: string | null;
  
  setJob: (id: string) => void;
  updateProgress: (data: any) => void;
  reset: () => void;
  setError: (error: string) => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  jobId: null,
  status: 'idle',
  progress: 0,
  message: '',
  result: null,
  error: null,
  
  setJob: (id: string) => set({ 
    jobId: id, 
    status: 'running', 
    progress: 0,
    message: '解析開始...',
    error: null 
  }),
  
  updateProgress: (data: any) => set({ 
    status: data.status, 
    progress: data.progress, 
    message: data.message,
    result: data.result || null,
    error: data.error || null,
  }),
  
  reset: () => set({ 
    jobId: null, 
    status: 'idle', 
    progress: 0, 
    message: '', 
    result: null,
    error: null 
  }),
  
  setError: (error: string) => set({ 
    status: 'failed', 
    error 
  }),
}));
