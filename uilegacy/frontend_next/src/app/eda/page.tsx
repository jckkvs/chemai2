// frontend_next/src/app/eda/page.tsx
'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { useQuery } from '@tanstack/react-query';
import { useChemAIStore } from '@/lib/store';
import { getEDAStats, getEDACorrelation, getEDADimReduction } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlertCircle, BarChart2, Network, Loader2 } from 'lucide-react';

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="h-[400px] w-full bg-slate-50 animate-pulse rounded-xl flex items-center justify-center text-slate-400">Loading visualization...</div>
});

export default function EDAPage() {
  const { sessionId, error } = useChemAIStore();
  const [activeTab, setActiveTab] = useState<'stats' | 'correlation' | 'pca' | 'tsne'>('stats');
  const [correlationMethod, setCorrelationMethod] = useState<'pearson' | 'spearman' | 'kendall'>('pearson');

  // Fetch EDA Data
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['edaStats', sessionId],
    queryFn: () => getEDAStats(sessionId!),
    enabled: !!sessionId,
  });

  const { data: correlation, isLoading: corrLoading } = useQuery({
    queryKey: ['edaCorrelation', sessionId, correlationMethod],
    queryFn: () => getEDACorrelation(sessionId!, correlationMethod),
    enabled: !!sessionId,
  });

  const { data: dimReduction, isLoading: dimLoading } = useQuery({
    queryKey: ['edaDimReduction', sessionId],
    queryFn: () => getEDADimReduction(sessionId!),
    enabled: !!sessionId,
  });

  if (!sessionId) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h2 className="text-xl font-semibold text-slate-600">セッションが有効ではありません。</h2>
        <p className="text-slate-500 mt-2">データをアップロードしてセッションを開始してください。</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">📊 探索的データ解析 (EDA)</h1>
          <p className="text-slate-500 mt-1">データの統計的特徴と分布を可視化します。</p>
        </div>
        <div className="flex flex-wrap gap-2 bg-slate-100 p-1 rounded-lg">
          <Button variant={activeTab === 'stats' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('stats')}><BarChart2 className="w-4 h-4 mr-2" />統計量</Button>
          <Button variant={activeTab === 'correlation' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('correlation')}><Network className="w-4 h-4 mr-2" />相関</Button>
          <Button variant={activeTab === 'pca' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('pca')}>PCA</Button>
          <Button variant={activeTab === 'tsne' ? 'default' : 'ghost'} size="sm" onClick={() => setActiveTab('tsne')}>t-SNE</Button>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-center gap-3">
          <AlertCircle className="w-5 h-5" /> {error}
        </div>
      )}

      {/* Stats Tab */}
      {activeTab === 'stats' && (
        <div className="grid gap-6">
          <Card className="overflow-hidden border-none shadow-md">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg font-bold">数値変数の統計サマリー</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {statsLoading ? (
                <div className="flex items-center justify-center p-20"><Loader2 className="w-8 h-8 animate-spin text-slate-300" /></div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 text-slate-500 uppercase text-xs font-bold tracking-wider">
                      <tr>
                        <th className="px-6 py-4">変数名</th>
                        <th className="px-4 py-4">Count</th>
                        <th className="px-4 py-4">Mean</th>
                        <th className="px-4 py-4">Std</th>
                        <th className="px-4 py-4">Min</th>
                        <th className="px-4 py-4">25%</th>
                        <th className="px-4 py-4">50%</th>
                        <th className="px-4 py-4">75%</th>
                        <th className="px-4 py-4">Max</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {stats?.stats?.map((s: any) => (
                        <tr key={s.column} className="hover:bg-slate-50 transition-colors">
                          <td className="px-6 py-4 font-semibold text-slate-900">{s.column}</td>
                          <td className="px-4 py-4 text-slate-600">{s.count}</td>
                          <td className="px-4 py-4 text-slate-600">{s.mean?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.std?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.min?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.q25?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.q50?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.q75?.toFixed(4)}</td>
                          <td className="px-4 py-4 text-slate-600">{s.max?.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Correlation Tab */}
      {activeTab === 'correlation' && (
        <Card className="border-none shadow-md overflow-hidden">
          <CardHeader className="bg-slate-50 border-b flex flex-row items-center justify-between">
            <CardTitle className="text-lg font-bold">相関ヒートマップ</CardTitle>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400 font-medium">Method:</span>
              <Select value={correlationMethod} onValueChange={(v: any) => setCorrelationMethod(v)}>
                <SelectTrigger className="w-32 h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="pearson">Pearson</SelectItem>
                  <SelectItem value="spearman">Spearman</SelectItem>
                  <SelectItem value="kendall">Kendall</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent className="p-6 flex justify-center bg-white">
            {corrLoading ? (
               <div className="flex items-center justify-center p-20"><Loader2 className="w-8 h-8 animate-spin text-slate-300" /></div>
            ) : correlation && correlation.columns.length > 0 ? (
              <Plot
                data={[{
                  z: correlation.matrix,
                  x: correlation.columns,
                  y: correlation.columns,
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  reversescale: true,
                  zmin: -1,
                  zmax: 1,
                  text: correlation.matrix.map((row: number[]) => row.map(val => val.toFixed(2))),
                  hoverinfo: 'text'
                }]}
                layout={{
                  width: Math.max(700, correlation.columns.length * 45),
                  height: Math.max(600, correlation.columns.length * 45),
                  margin: { l: 150, r: 50, t: 50, b: 150 },
                  xaxis: { tickangle: -45, automargin: true },
                  yaxis: { automargin: true }
                }}
                config={{ responsive: true, displayModeBar: false }}
              />
            ) : (
              <div className="p-20 text-slate-400">表示可能な相関データがありません。</div>
            )}
          </CardContent>
        </Card>
      )}

      {/* PCA/t-SNE Tabs */}
      {(activeTab === 'pca' || activeTab === 'tsne') && (
        <Card className="border-none shadow-md overflow-hidden">
          <CardHeader className="bg-slate-50 border-b">
            <CardTitle className="text-lg font-bold">{activeTab === 'pca' ? 'PCA (主成分分析)' : 't-SNE (非線形次元削減)'}</CardTitle>
          </CardHeader>
          <CardContent className="p-6 flex flex-col items-center bg-white">
            {dimLoading ? (
               <div className="flex items-center justify-center p-20"><Loader2 className="w-8 h-8 animate-spin text-slate-300" /></div>
            ) : dimReduction && dimReduction.pca ? (
              <>
                <Plot
                  data={[{
                    x: activeTab === 'pca' ? dimReduction.pca.map((p: any) => p[0]) : dimReduction.tsne.map((p: any) => p[0]),
                    y: activeTab === 'pca' ? dimReduction.pca.map((p: any) => p[1]) : dimReduction.tsne.map((p: any) => p[1]),
                    mode: 'markers',
                    type: 'scatter',
                    marker: { 
                      size: 10, 
                      color: activeTab === 'pca' ? dimReduction.pca.map((p: any) => p[0]) : '#6366f1', 
                      colorscale: 'Viridis',
                      line: { width: 1, color: 'white' },
                      opacity: 0.8
                    }
                  }]}
                  layout={{
                    width: 800,
                    height: 600,
                    margin: { l: 50, r: 50, t: 30, b: 50 },
                    xaxis: { title: activeTab === 'pca' ? 'PC1' : 't-SNE 1', gridcolor: '#f1f5f9' },
                    yaxis: { title: activeTab === 'pca' ? 'PC2' : 't-SNE 2', gridcolor: '#f1f5f9' },
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                  }}
                  config={{ responsive: true }}
                />
                {activeTab === 'pca' && dimReduction.explained_variance && (
                  <div className="mt-6 flex gap-6">
                    {dimReduction.explained_variance.map((v: number, i: number) => (
                      <div key={i} className="text-center">
                        <span className="text-xs text-slate-400 font-bold uppercase block mb-1">PC{i+1} Variance</span>
                        <span className="text-xl font-bold text-slate-700">{(v * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="p-20 text-slate-400">十分なデータがないため、次元削減を実行できません。</div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
