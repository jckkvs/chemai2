// frontend_next/src/app/page.tsx
'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useChemAIStore } from '@/lib/store';
import { initSession, healthCheck } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Activity, Settings, BarChart3, FlaskConical, ArrowRight, CheckCircle2 } from 'lucide-react';

export default function Home() {
  const router = useRouter();
  const { sessionId, setSessionId, error, setError } = useChemAIStore();

  useEffect(() => {
    const initialize = async () => {
      if (!sessionId) {
        try {
          const id = await initSession();
          setSessionId(id);
        } catch (err) {
          console.error('Failed to initialize session:', err);
          setError('セッションの初期化に失敗しました。サーバーの状態を確認してください。');
        }
      }
      
      // Health check
      try {
        await healthCheck();
      } catch (err) {
        console.warn('Backend health check failed:', err);
      }
    };
    
    initialize();
  }, [sessionId, setSessionId, setError]);

  const features = [
    {
      icon: Upload,
      title: 'データ読込',
      description: 'CSV/Excel ファイルまたはベンチマークデータセットをアップロード',
      href: '/data',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: BarChart3,
      title: 'EDA・可視化',
      description: '統計解析、相関分析、次元削減、クラスタリング',
      href: '/eda',
      color: 'from-emerald-500 to-teal-500',
    },
    {
      icon: Settings,
      title: 'パイプライン構築',
      description: '前処理・特徴量・モデル設定を自由にカスタマイズ',
      href: '/pipeline',
      color: 'from-violet-500 to-purple-500',
    },
    {
      icon: Activity,
      title: '解析実行・結果',
      description: 'モデル学習、評価、解釈可能性分析',
      href: '/results',
      color: 'from-orange-500 to-amber-500',
    },
  ];

  return (
    <div className="container mx-auto px-4 py-12">
      {/* Hero Section */}
      <section className="text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-xs font-semibold mb-6 border border-indigo-100">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
          </span>
          ChemAI Nexus v2.0.0 Now Active
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold text-slate-900 mb-6 tracking-tight">
          Accelerate Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-blue-500">Chemistry AI</span>
        </h1>
        <p className="text-xl text-slate-600 max-w-3xl mx-auto leading-relaxed">
          化学構造データとテーブルデータを統合し、高度な機械学習モデルを直感的に構築。
          ワンクリックの自動解析から、専門家向けの高度なチューニングまで対応します。
        </p>
        
        <div className="mt-10 flex flex-wrap justify-center gap-4">
          <Button 
            size="lg" 
            onClick={() => router.push('/data')}
            className="bg-indigo-600 hover:bg-indigo-700 h-12 px-8 text-lg font-bold shadow-xl shadow-indigo-100"
          >
            Start Project <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
          <Button 
            variant="outline" 
            size="lg"
            className="h-12 px-8 text-lg font-semibold"
          >
            Documentation
          </Button>
        </div>
      </section>

      {/* Feature Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
        {features.map((feature) => (
          <Card 
            key={feature.title}
            className="group relative overflow-hidden cursor-pointer border-slate-200 transition-all duration-300 hover:border-indigo-200 hover:shadow-2xl hover:shadow-indigo-50"
            onClick={() => router.push(feature.href)}
          >
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <feature.icon size={80} />
            </div>
            <CardHeader className="pb-2">
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 text-white shadow-lg group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-7 h-7" />
              </div>
              <CardTitle className="text-xl font-bold">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">{feature.description}</p>
              <div className="flex items-center text-indigo-600 text-sm font-bold group-hover:translate-x-1 transition-transform">
                Go to Module <ArrowRight className="ml-1 w-4 h-4" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Status & Actions */}
      <div className="grid lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-2 border-slate-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="text-indigo-600 w-5 h-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <p className="text-xs font-bold text-slate-400 uppercase mb-1">Backend Connection</p>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <p className="font-bold text-slate-700">Healthy</p>
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <p className="text-xs font-bold text-slate-400 uppercase mb-1">Session Identity</p>
                <p className="font-mono text-xs font-medium text-slate-600 truncate">
                  {sessionId || 'Initializing...'}
                </p>
              </div>
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <p className="text-xs font-bold text-slate-400 uppercase mb-1">Active Projects</p>
                <p className="font-bold text-slate-700">0 Total</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical className="text-indigo-600 w-5 h-5" />
              SMILES Support
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500 mb-6 leading-relaxed">
              RDKit 統合により、SMILES 文字列から化学記述子、指紋、3D 特徴量を自動生成。
              深層学習ポテンシャルとの連携もサポートしています。
            </p>
            <Button variant="outline" className="w-full font-bold">
              Explore Chem Features
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Error Message */}
      {error && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 w-full max-w-md animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="bg-red-900 text-white px-6 py-4 rounded-2xl shadow-2xl flex items-center gap-3">
            <div className="p-2 bg-red-800 rounded-lg">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <p className="font-medium text-sm">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
