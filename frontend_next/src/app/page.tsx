// frontend_next/src/app/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { Upload, Activity, Settings, Play, Loader2, CheckCircle } from 'lucide-react';
import { initSession, uploadData, getModels, runPipeline } from '@/lib/api';

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [models, setModels] = useState<any[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle');
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    const setup = async () => {
      try {
        const id = await initSession();
        setSessionId(id);
        const modelList = await getModels('regression');
        setModels(modelList);
      } catch (error) {
        console.error('Failed to initialize session or fetch models:', error);
      }
    };
    setup();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus('uploading');
    try {
      await uploadData(file);
      setUploadStatus('success');
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadStatus('error');
    }
  };

  const handleRunPipeline = async () => {
    if (selectedModels.length === 0) return;
    setPipelineStatus('running');
    try {
      const res = await runPipeline({
        cv_folds: 5,
        selected_models: selectedModels,
        num_scaler: 'standard',
      });
      setResult(res);
      setPipelineStatus('completed');
    } catch (error) {
      console.error('Pipeline failed:', error);
      setPipelineStatus('error');
    }
  };

  const toggleModel = (key: string) => {
    setSelectedModels(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="mb-10">
          <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-8 h-8 text-blue-600" />
            ChemAI Nexus
          </h1>
          <p className="text-slate-500 mt-2">
            Session: {sessionId ? <span className="text-green-600 font-mono">{sessionId.slice(0, 8)}...</span> : 'Initializing...'}
          </p>
        </header>

        {/* Section 1: Data Upload */}
        <section className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-slate-600" />
            1. データ読込
          </h2>
          <div className="flex items-center gap-4">
            <label className="flex-1 border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:bg-slate-50 cursor-pointer transition">
              <input type="file" accept=".csv,.xlsx" onChange={handleFileChange} className="hidden" />
              {file ? (
                <span className="text-blue-600 font-medium">{file.name}</span>
              ) : (
                <span className="text-slate-500">CSV または Excel ファイルを選択</span>
              )}
            </label>
            <button
              onClick={handleUpload}
              disabled={!file || uploadStatus === 'uploading'}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {uploadStatus === 'uploading' ? <Loader2 className="w-5 h-5 animate-spin" /> : 'アップロード'}
            </button>
          </div>
          {uploadStatus === 'success' && <p className="mt-2 text-green-600 text-sm">✅ アップロード完了</p>}
          {uploadStatus === 'error' && <p className="mt-2 text-red-600 text-sm">❌ アップロードに失敗しました</p>}
        </section>

        {/* Section 2: Model Selection */}
        <section className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-slate-600" />
            2. モデル選択
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {models.map((model) => (
              <label key={model.key} className={`p-3 border rounded-lg cursor-pointer transition flex items-center gap-2 ${selectedModels.includes(model.key) ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:border-blue-300'}`}>
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.key)}
                  onChange={() => toggleModel(model.key)}
                  className="w-4 h-4 text-blue-600 rounded"
                />
                <div>
                  <div className="font-medium text-sm">{model.name}</div>
                  <div className="text-xs text-slate-500">{model.key}</div>
                </div>
              </label>
            ))}
          </div>
        </section>

        {/* Section 3: Execution */}
        <section className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Play className="w-5 h-5 text-slate-600" />
            3. 解析実行
          </h2>
          <button
            onClick={handleRunPipeline}
            disabled={pipelineStatus === 'running' || selectedModels.length === 0}
            className="w-full bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            {pipelineStatus === 'running' ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" /> 実行中...
              </>
            ) : (
              '解析を開始する'
            )}
          </button>

          {result && pipelineStatus === 'completed' && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="font-semibold text-green-800 flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5" /> 解析完了
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-slate-500">最良モデル:</span>
                  <span className="ml-2 font-medium text-slate-900">{result.best_model}</span>
                </div>
                <div>
                  <span className="text-slate-500">スコア (R²):</span>
                  <span className="ml-2 font-medium text-slate-900">{result.score}</span>
                </div>
              </div>
              {result.feature_importances && (
                <div className="mt-4">
                  <span className="text-slate-500 text-sm">特徴量重要度 Top 3:</span>
                  <ul className="mt-2 space-y-1">
                    {result.feature_importances.slice(0, 3).map((imp: any, idx: number) => (
                      <li key={idx} className="flex justify-between text-sm bg-white p-2 rounded">
                        <span>{imp.name}</span>
                        <span className="font-mono">{imp.value.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>

      </div>
    </main>
  );
}
