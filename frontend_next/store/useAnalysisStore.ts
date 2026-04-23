import { create } from 'zustand';

type AnalysisState = {
  jobId: string | null;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'pending';
  progress: number;
  message: string;
  result: any | null;
  setJob: (id: string) => void;
  updateProgress: (data: any) => void;
  reset: () => void;
};

export const useAnalysisStore = create<AnalysisState>((set) => ({
  jobId: null,
  status: 'idle',
  progress: 0,
  message: '',
  result: null,
  setJob: (id) => set({ jobId: id, status: 'running', progress: 0 }),
  updateProgress: (data) => set({ 
    status: data.status, 
    progress: data.progress, 
    message: data.message,
    result: data.result || null 
  }),
  reset: () => set({ jobId: null, status: 'idle', progress: 0, message: '', result: null }),
}));
