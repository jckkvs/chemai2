// frontend_next/src/app/results/page.tsx
'use client';

import dynamic from 'next/dynamic';
import { useQuery } from '@tanstack/react-query';
import { useChemAIStore } from '@/lib/store';
import { getResults } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, CheckCircle, Loader2, Info } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="h-[300px] w-full bg-slate-50 animate-pulse rounded-xl flex items-center justify-center text-slate-400">Loading result visualization...</div>
});

export default function ResultsPage() {
  const { sessionId, error: storeError } = useChemAIStore();
  
  const { data: result, isLoading, error } = useQuery({
    queryKey: ['results', sessionId],
    queryFn: () => getResults(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) => {
      // Access the data from the query state
      const data = query.state.data;
      return (data && data.status === 'running') ? 2000 : false;
    },
  });

  if (!sessionId) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h2 className="text-xl font-semibold text-slate-600">セッションが有効ではありません。</h2>
        <p className="text-slate-500 mt-2">データをアップロードして解析を実行してください。</p>
      </div>
    );
  }

  if (isLoading) return <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
    <Loader2 className="animate-spin w-12 h-12 text-blue-500" />
    <p className="text-slate-500 animate-pulse">解析結果を読み込んでいます...</p>
  </div>;

  const displayError = error instanceof Error ? error.message : (error || storeError);

  if (displayError) return (
    <div className="container mx-auto px-4 py-8">
      <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-center gap-3">
        <AlertCircle className="w-5 h-5" /> {String(displayError)}
      </div>
    </div>
  );

  if (!result || result.status === 'pending') return (
    <div className="container mx-auto px-4 py-12 text-center">
      <div className="max-w-md mx-auto bg-slate-50 p-8 rounded-2xl border border-slate-100">
        <Info className="w-12 h-12 text-slate-300 mx-auto mb-4" />
        <h3 className="text-lg font-bold text-slate-900">解析が完了していません。</h3>
        <p className="text-slate-500 mt-2 mb-6">パイプラインビルダーで「解析実行」ボタンをクリックしてください。</p>
        <button 
          onClick={() => window.location.href = '/pipeline'}
          className="px-6 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors"
        >
          パイプラインビルダーへ
        </button>
      </div>
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">📈 解析結果・解釈可能性</h1>
          <p className="text-slate-500 mt-1">構築されたモデルの性能と特徴量の寄与を分析します。</p>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold shadow-sm border ${
          result.status === 'completed' ? 'bg-green-50 text-green-700 border-green-100' : 'bg-blue-50 text-blue-700 border-blue-100'
        }`}>
          {result.status === 'completed' ? <CheckCircle className="w-4 h-4" /> : <Loader2 className="w-4 h-4 animate-spin" />}
          {result.status === 'completed' ? '解析完了' : '実行中...'}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="border-none shadow-md bg-white">
          <CardHeader className="pb-2"><CardTitle className="text-xs text-slate-400 font-bold uppercase tracking-wider">アルゴリズム</CardTitle></CardHeader>
          <CardContent><p className="text-2xl font-black text-slate-900">{result.best_model || '-'}</p></CardContent>
        </Card>
        <Card className="border-none shadow-md bg-white">
          <CardHeader className="pb-2"><CardTitle className="text-xs text-slate-400 font-bold uppercase tracking-wider">CV Score (Mean)</CardTitle></CardHeader>
          <CardContent><p className="text-3xl font-black text-blue-600">{result.score?.toFixed(4) || '-'}</p></CardContent>
        </Card>
        <Card className="border-none shadow-md bg-white">
          <CardHeader className="pb-2"><CardTitle className="text-xs text-slate-400 font-bold uppercase tracking-wider">Features</CardTitle></CardHeader>
          <CardContent><p className="text-3xl font-black text-slate-900">{result.metadata?.n_features || '-'}</p></CardContent>
        </Card>
        <Card className="border-none shadow-md bg-white">
          <CardHeader className="pb-2"><CardTitle className="text-xs text-slate-400 font-bold uppercase tracking-wider">制約適用数</CardTitle></CardHeader>
          <CardContent><p className="text-3xl font-black text-slate-900">{result.metadata?.constraints_applied || 0}</p></CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Feature Importance / SHAP */}
        <Card className="border-none shadow-md bg-white overflow-hidden">
          <CardHeader className="bg-slate-50 border-b"><CardTitle className="text-lg font-bold">特徴量重要度 / SHAP 値</CardTitle></CardHeader>
          <CardContent className="p-6 flex justify-center">
            {result.feature_importances && result.feature_importances.length > 0 ? (
              <Plot
                data={[{
                  y: result.feature_importances.map(i => i.name).reverse(),
                  x: result.feature_importances.map(i => i.value).reverse(),
                  type: 'bar',
                  orientation: 'h',
                  marker: { 
                    color: '#4f46e5',
                    line: { width: 1, color: 'white' }
                  }
                }]}
                layout={{
                  width: 500,
                  height: Math.max(400, result.feature_importances.length * 35),
                  margin: { l: 150, r: 30, t: 30, b: 50 },
                  xaxis: { title: 'Importance', gridcolor: '#f1f5f9' },
                  yaxis: { automargin: true },
                  plot_bgcolor: 'white',
                  paper_bgcolor: 'white',
                }}
                config={{ responsive: true, displayModeBar: false }}
              />
            ) : (
              <div className="p-20 text-slate-400 text-center">
                重要度データを取得できませんでした。
              </div>
            )}
          </CardContent>
        </Card>

        {/* CV Scores Distribution */}
        <Card className="border-none shadow-md bg-white overflow-hidden">
          <CardHeader className="bg-slate-50 border-b"><CardTitle className="text-lg font-bold">交差検証スコアの詳細 (Folds)</CardTitle></CardHeader>
          <CardContent className="p-6 flex flex-col items-center">
            {result.cv_scores ? (
              <>
                <Plot
                  data={[{
                    y: result.cv_scores,
                    x: result.cv_scores.map((_, i) => `Fold ${i + 1}`),
                    type: 'bar',
                    marker: { 
                      color: '#10b981',
                      line: { width: 1, color: 'white' }
                    }
                  }]}
                  layout={{ 
                    width: 500, 
                    height: 400, 
                    yaxis: { title: 'Score', gridcolor: '#f1f5f9' },
                    xaxis: { gridcolor: 'transparent' },
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                />
                <div className="mt-4 grid grid-cols-2 gap-8 w-full max-w-xs">
                   <div className="text-center">
                    <span className="text-[10px] text-slate-400 font-bold uppercase block">Variance</span>
                    <span className="text-lg font-bold text-slate-700">
                      {Math.sqrt(result.cv_scores.reduce((a, b) => a + Math.pow(b - (result.score || 0), 2), 0) / result.cv_scores.length).toFixed(6)}
                    </span>
                  </div>
                  <div className="text-center">
                    <span className="text-[10px] text-slate-400 font-bold uppercase block">Min / Max</span>
                    <span className="text-lg font-bold text-slate-700">
                      {Math.min(...result.cv_scores).toFixed(3)} / {Math.max(...result.cv_scores).toFixed(3)}
                    </span>
                  </div>
                </div>
              </>
            ) : (
              <div className="p-20 text-slate-400">CVデータがありません。</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* SHAP Detail Mock (Future Phase) */}
      <Card className="border-none shadow-md bg-slate-900 text-white overflow-hidden">
        <CardContent className="p-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2">
            <h3 className="text-xl font-bold">🧪 次のステップ: SHAP 局所説明</h3>
            <p className="text-slate-400 text-sm max-w-xl">
              個別のサンプルごとの寄与を分析する準備ができています。SHAP Beeswarm Plot や Force Plot を使用して、特定の分子構造がなぜその予測値になったのかを詳細に分析しましょう。
            </p>
          </div>
          <button className="px-6 py-3 bg-white text-slate-900 rounded-xl font-bold hover:bg-slate-100 transition-colors whitespace-nowrap">
            詳細な解釈を開始
          </button>
        </CardContent>
      </Card>
    </div>
  );
}
