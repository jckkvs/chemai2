// frontend_next/src/app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useChemAIStore } from '@/lib/store';
import { initSession, healthCheck } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Activity, Settings, BarChart3, FlaskConical, Database } from 'lucide-react';

export default function Home() {
  const router = useRouter();
  const { sessionId, setSessionId, error } = useChemAIStore();
  const [backendStatus, setBackendStatus] = useState<'checking' | 'healthy' | 'error'>('checking');

  useEffect(() => {
    const initialize = async () => {
      // Initialize session
      if (!sessionId) {
        try {
          const id = await initSession();
          setSessionId(id);
        } catch (err) {
          console.error('Failed to initialize session:', err);
        }
      }
      
      // Health check
      try {
        const health = await healthCheck();
        setBackendStatus('healthy');
        console.log('Backend health:', health);
      } catch (err) {
        setBackendStatus('error');
        console.warn('Backend health check failed:', err);
      }
    };
    
    initialize();
  }, [sessionId, setSessionId]);

  const features = [
    {
      icon: Upload,
      title: 'データ読込',
      description: 'CSV/Excel ファイルまたはベンチマークデータセットをアップロード。SMILES 列を自動検出。',
      href: '/data',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: BarChart3,
      title: 'EDA・可視化',
      description: '統計解析、相関分析、PCA/t-SNE/UMAP による次元削減、クラスタリング。',
      href: '/eda',
      color: 'from-emerald-500 to-teal-500',
    },
    {
      icon: Settings,
      title: 'パイプライン構築',
      description: '前処理・特徴量生成・特徴量選択・モデル設定を自由にカスタマイズ。単調性制約も設定可能。',
      href: '/pipeline',
      color: 'from-violet-500 to-purple-500',
    },
    {
      icon: Activity,
      title: '解析実行・結果',
      description: 'モデル学習、交差検証、評価指標、SHAP/PDP による解釈可能性分析。',
      href: '/results',
      color: 'from-orange-500 to-amber-500',
    },
    {
      icon: FlaskConical,
      title: 'SMILES 特徴量',
      description: 'RDKit/XTB/COSMO-RS 等による化学記述子計算。プラグインで拡張可能。',
      href: '/pipeline#features',
      color: 'from-pink-500 to-rose-500',
    },
    {
      icon: Database,
      title: 'MLOps',
      description: '実験管理 (MLflow)、モデルバージョン管理、デプロイ準備。',
      href: '/mlops',
      color: 'from-indigo-500 to-blue-500',
    },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Hero Section */}
      <section className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
          ChemAI Nexus
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          化学構造データと機械学習を統合した解析プラットフォーム。
          初心者向けワンクリック解析から、専門家向け完全制御まで。
        </p>
        
        <div className="mt-6 flex items-center justify-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
            backendStatus === 'healthy' ? 'bg-green-50 text-green-700' :
            backendStatus === 'error' ? 'bg-red-50 text-red-700' :
            'bg-amber-50 text-amber-700'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              backendStatus === 'healthy' ? 'bg-green-500 animate-pulse' :
              backendStatus === 'error' ? 'bg-red-500' :
              'bg-amber-500 animate-pulse'
            }`}></span>
            {backendStatus === 'healthy' ? 'Backend: Connected' :
             backendStatus === 'error' ? 'Backend: Disconnected' :
             'Backend: Checking...'}
          </div>
          
          {sessionId && (
            <div className="flex items-center gap-2 px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
              Session: <code className="font-mono">{sessionId.slice(0, 8)}...</code>
            </div>
          )}
        </div>
      </section>

      {/* Quick Start Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        {features.map((feature) => (
          <Card 
            key={feature.title}
            className="group cursor-pointer hover:shadow-lg transition-all duration-200 border-slate-200 hover:border-slate-300"
            onClick={() => router.push(feature.href)}
          >
            <CardHeader className="pb-2">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-3 group-hover:scale-105 transition-transform`}>
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <CardTitle className="text-lg">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-slate-600 text-sm">{feature.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Status & Quick Actions */}
      <section className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">システム状態</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">バックエンド</span>
              <span className={`flex items-center gap-1 ${
                backendStatus === 'healthy' ? 'text-green-600' :
                backendStatus === 'error' ? 'text-red-600' : 'text-amber-600'
              }`}>
                <span className={`w-2 h-2 rounded-full ${
                  backendStatus === 'healthy' ? 'bg-green-500' :
                  backendStatus === 'error' ? 'bg-red-500' : 'bg-amber-500 animate-pulse'
                }`}></span>
                {backendStatus === 'healthy' ? 'Connected' :
                 backendStatus === 'error' ? 'Disconnected' : 'Checking...'}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">セッション</span>
              <span className={sessionId ? 'text-green-600' : 'text-amber-600'}>
                {sessionId ? 'Active' : 'Initializing...'}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">データ</span>
              <span className="text-slate-400">Not loaded</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">特徴量エンジン</span>
              <span className="text-slate-400">Loading...</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">クイックアクション</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button 
              variant="outline" 
              className="w-full justify-start"
              onClick={() => router.push('/data')}
            >
              <Upload className="w-4 h-4 mr-2" />
              データをアップロード
            </Button>
            <Button 
              variant="outline" 
              className="w-full justify-start"
              onClick={() => router.push('/pipeline')}
            >
              <Settings className="w-4 h-4 mr-2" />
              新規パイプライン作成
            </Button>
            <Button 
              variant="outline" 
              className="w-full justify-start"
              onClick={() => router.push('/data#benchmarks')}
            >
              <Database className="w-4 h-4 mr-2" />
              ベンチマークデータを読み込み
            </Button>
            <Button 
              variant="outline" 
              className="w-full justify-start"
              disabled
            >
              <FlaskConical className="w-4 h-4 mr-2" />
              SMILES 特徴量を計算
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* Feature Highlights */}
      <section className="mt-12">
        <h2 className="text-xl font-semibold text-slate-900 mb-4">主な機能</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-white rounded-lg border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-2">🔄 柔軟なパイプライン</h3>
            <p className="text-sm text-slate-600">
              sklearn の制限を超えたカスタムパイプライン。メタ特徴量・SMILES 特徴量・単調性制約をネイティブサポート。
            </p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-2">🔌 プラグインアーキテクチャ</h3>
            <p className="text-sm text-slate-600">
              特徴量計算エンジンを独立した .py として追加可能。動的 UI 生成で設定も簡単。
            </p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-2">⚖️ 制約付き学習</h3>
            <p className="text-sm text-slate-600">
              単調性・線形性制約を「強い/弱い」「±nσ範囲」で柔軟に適用。ドメイン知識をモデルに反映。
            </p>
          </div>
        </div>
      </section>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 text-sm">⚠️ {error}</p>
        </div>
      )}
    </div>
  );
}
