"use client";
import { useState } from 'react';
import { DataUploader } from '../components/ui/DataUploader';
import { AnalysisForm } from '../components/ui/AnalysisForm';
import { useAnalysis } from '../hooks/useAnalysis';

export default function HomePage() {
  const [dataLoaded, setDataLoaded] = useState(false);
  const [config, setConfig] = useState({
    cv_folds: 5,
    num_scaler: 'standard',
    selected_models: ['RandomForest', 'XGBoost', 'LightGBM'],
  });

  const { 
    status, 
    progress, 
    message, 
    result, 
    error, 
    startAnalysis, 
    reset 
  } = useAnalysis();

  const handleDataLoaded = (data: any) => {
    setDataLoaded(true);
    // データ読み込み完了処理
  };

  const handleStart = () => {
    // 実際のファイルデータは本来 state に保持しておく必要がある
    // ここでは簡略化のため config のみ渡す
    startAnalysis(config, null); 
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* ヘッダー */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            ChemAI Nexus
          </h1>
          <p className="text-slate-400">Next.js + FastAPI Migration</p>
        </div>

        {/* ステップ 1: データアップロード */}
        {!dataLoaded && (
          <div className="animate-fade-in">
            <h2 className="text-xl font-semibold text-white mb-4">Step 1: データの読み込み</h2>
            <DataUploader onDataLoaded={handleDataLoaded} />
          </div>
        )}

        {/* ステップ 2: 設定と解析実行 */}
        {dataLoaded && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-slide-up">
            <div className="lg:col-span-2">
              <AnalysisForm 
                config={config}
                isRunning={status === 'running'}
                onConfigChange={(key, value) => setConfig(prev => ({ ...prev, [key]: value }))}
                onStart={handleStart}
              />
            </div>
            
            {/* 進捗パネル */}
            {(status === 'running' || result || error) && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                <h3 className="font-bold text-white mb-4">実行状況</h3>
                
                {status === 'running' && (
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm text-slate-400">
                      <span>{message}</span>
                      <span>{Math.round(progress * 100)}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-2">
                      <div 
                        className="bg-cyan-500 h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${progress * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {error && (
                  <div className="text-red-400 text-sm p-3 bg-red-500/10 rounded border border-red-500/20">
                    {error}
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    <div className="text-green-400 font-medium">✅ 解析完了</div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">Best Model</span>
                        <span className="text-white font-bold">{result.best_model}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">R² Score</span>
                        <span className="text-white font-bold">{result.metrics?.r2}</span>
                      </div>
                    </div>
                    <button 
                      onClick={reset}
                      className="mt-4 w-full py-2 text-sm text-slate-400 hover:text-white border border-slate-600 hover:border-slate-400 rounded transition-colors"
                    >
                      別の解析を実行
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
