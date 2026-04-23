import { useState, useCallback, useEffect } from 'react';
import { api } from '../lib/api';
import axios from 'axios';

interface AnalysisState {
  jobId: string | null;
  status: 'idle' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  result: any | null;
  error: string | null;
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    jobId: null,
    status: 'idle',
    progress: 0,
    message: '',
    result: null,
    error: null,
  });

  // WebSocket 接続管理
  useEffect(() => {
    if (!state.jobId || state.status !== 'running') return;

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/api/analysis/ws/progress/${state.jobId}`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      setState(prev => ({
        ...prev,
        progress: data.progress,
        message: data.message,
        status: data.status,
        result: data.status === 'completed' ? data.result : prev.result,
        error: data.status === 'failed' ? data.error : null
      }));
    };

    socket.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      socket.close();
    };
  }, [state.jobId, state.status]);

  const startAnalysis = useCallback(async (config: any, df: any) => {
    setState(prev => ({ ...prev, status: 'running', progress: 0, message: '初期化中...', error: null }));

    try {
      // ファイルデータを FormData に変換して送信（例）
      // const formData = new FormData();
      // formData.append('file', df);
      // formData.append('config', JSON.stringify(config));
      
      // ここでは簡易的に config のみ送信
      const { data } = await api.post('/api/analysis/run', { config });
      
      setState(prev => ({ ...prev, jobId: data.job_id }));
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        status: 'failed',
        error: err.response?.data?.detail || '解析リクエストに失敗しました'
      }));
    }
  }, []);

  const cancelAnalysis = useCallback(async () => {
    if (!state.jobId) return;
    try {
      await api.delete(`/api/analysis/cancel/${state.jobId}`);
      setState(prev => ({ ...prev, status: 'idle', message: 'キャンセルされました' }));
    } catch (err) {
      console.error('Cancel failed', err);
    }
  }, [state.jobId]);

  const reset = useCallback(() => {
    setState({
      jobId: null,
      status: 'idle',
      progress: 0,
      message: '',
      result: null,
      error: null,
    });
  }, []);

  return {
    ...state,
    startAnalysis,
    cancelAnalysis,
    reset,
  };
}
