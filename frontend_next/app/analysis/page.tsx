"use client";
import { useState } from 'react';
import { useAnalysisStore } from '@/store/useAnalysisStore';
import { api } from '@/lib/api';

export default function AnalysisPage() {
  const { jobId, status, progress, message, setJob, updateProgress, reset } = useAnalysisStore();
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    reset();
    try {
      const { data } = await api.post('/api/analysis/run', {
        config: { 
          cv_folds: 5, 
          models: ['rf', 'xgb'], 
          scaler: 'standard',
          df_json: '[]' // 実際はデータのJSONを送る
        }
      });
      setJob(data.job_id);
      
      // WebSocket 監視開始
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//localhost:8000/api/analysis/ws/progress/${data.job_id}`);
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        updateProgress(data);
        if (['completed', 'failed'].includes(data.status)) ws.close();
      };
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">機械学習パイプライン</h1>
      <button 
        onClick={handleStart} 
        disabled={loading || status === 'running'}
        className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-all"
      >
        {loading ? '送信中...' : status === 'running' ? '実行中...' : '🚀 解析開始'}
      </button>

      {status !== 'idle' && (
        <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
          <div className="flex justify-between text-sm mb-2 text-white">
            <span>{message}</span>
            <span>{Math.round(progress * 100)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div 
              className="bg-blue-500 h-2.5 rounded-full transition-all duration-300" 
              style={{ width: `${progress * 100}%` }}
            ></div>
          </div>
          {status === 'completed' && (
            <div className="mt-4 text-green-400 font-semibold">✅ 解析完了。結果タブへ移動してください。</div>
          )}
          {status === 'failed' && (
            <div className="mt-4 text-red-400 font-semibold">❌ 解析失敗。エラー内容を確認してください。</div>
          )}
        </div>
      )}
    </div>
  );
}
