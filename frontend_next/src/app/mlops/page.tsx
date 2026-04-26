// frontend_next/src/app/mlops/page.tsx
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { FlaskConical, Download, Eye, GitBranch, Trash2, Calendar, Database, Target } from 'lucide-react';
import dynamic from 'next/dynamic';
import { listExperiments } from '@/lib/api';
import { useChemAIStore } from '@/lib/store';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function MLOpsPage() {
  const { sessionId } = useChemAIStore();
  const [selectedRun, setSelectedRun] = useState<any>(null);

  const { data: experiments, isLoading } = useQuery({
    queryKey: ['experiments', sessionId],
    queryFn: () => listExperiments(sessionId!),
    enabled: !!sessionId
  });

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">🧪 MLOps & Experiment Tracking</h1>
          <p className="text-slate-500 mt-1">実験履歴の管理、モデルの比較、メトリクスの追跡を行います。</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="rounded-xl shadow-sm hover:shadow-md transition-all">
            <GitBranch className="w-4 h-4 mr-2" />モデル比較
          </Button>
          <Button variant="outline" className="rounded-xl shadow-sm hover:shadow-md transition-all">
            <Download className="w-4 h-4 mr-2" />エクスポート
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Experiment History Table */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="border-none shadow-xl overflow-hidden rounded-2xl">
            <CardHeader className="bg-slate-50/50 border-b">
              <CardTitle className="text-xl flex items-center gap-2">
                <Database className="w-5 h-5 text-indigo-500" />
                実験履歴 (Local History)
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader className="bg-slate-50">
                  <TableRow>
                    <TableHead className="w-[200px] font-bold">実験名 / ID</TableHead>
                    <TableHead className="font-bold">アルゴリズム</TableHead>
                    <TableHead className="font-bold">ステータス</TableHead>
                    <TableHead className="font-bold text-right">CV Score</TableHead>
                    <TableHead className="font-bold">作成日時</TableHead>
                    <TableHead className="w-[60px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    <TableRow><TableCell colSpan={6} className="text-center py-20 text-slate-400">読み込み中...</TableCell></TableRow>
                  ) : experiments && experiments.length > 0 ? (
                    experiments.map((exp: any) => (
                      <TableRow 
                        key={exp.id} 
                        className={`cursor-pointer transition-colors ${selectedRun?.id === exp.id ? 'bg-indigo-50/50' : 'hover:bg-slate-50'}`} 
                        onClick={() => setSelectedRun(exp)}
                      >
                        <TableCell>
                          <div className="font-bold text-slate-900 truncate max-w-[180px]">{exp.name}</div>
                          <div className="text-[10px] font-mono text-slate-400 truncate">{exp.id}</div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="bg-white border-slate-200 text-slate-600 font-medium">{exp.model}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge 
                            variant={exp.status === 'completed' ? 'success' : exp.status === 'failed' ? 'destructive' : 'secondary'}
                          >
                            {exp.status === 'completed' ? '完了' : exp.status === 'failed' ? '失敗' : '実行中'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-black text-indigo-600">
                          {exp.score ? exp.score.toFixed(4) : '-'}
                        </TableCell>
                        <TableCell className="text-slate-400 text-xs">
                          <div className="flex items-center gap-1"><Calendar className="w-3 h-3" /> {new Date(exp.created_at).toLocaleDateString()}</div>
                          <div className="ml-4">{new Date(exp.created_at).toLocaleTimeString()}</div>
                        </TableCell>
                        <TableCell>
                          <Button variant="ghost" size="icon" className="text-slate-300 hover:text-indigo-600">
                            <Eye className="w-4 h-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center py-24 text-slate-400">
                        <FlaskConical className="w-12 h-12 mx-auto mb-4 opacity-20" />
                        <p>実験履歴がありません。パイプラインを実行して結果を保存してください。</p>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        {/* Selected Run Details */}
        <div className="space-y-6">
          <Card className="border-none shadow-2xl rounded-3xl overflow-hidden h-full flex flex-col">
            <CardHeader className="bg-indigo-600 text-white p-8">
              <div className="flex justify-between items-start">
                <Badge className="bg-indigo-400 border-none text-white font-bold">{selectedRun ? selectedRun.status.toUpperCase() : 'NO SELECTION'}</Badge>
                {selectedRun && <Button variant="ghost" size="icon" className="text-indigo-200 hover:text-white hover:bg-indigo-500"><Trash2 className="w-4 h-4" /></Button>}
              </div>
              <CardTitle className="text-2xl font-black mt-4">
                {selectedRun ? selectedRun.name : 'Run Details'}
              </CardTitle>
              <CardDescription className="text-indigo-100 opacity-70">
                {selectedRun ? `${selectedRun.model} | Created by Nexus Engine` : '実験を選択すると詳細が表示されます'}
              </CardDescription>
            </CardHeader>
            <CardContent className="p-8 flex-1 space-y-8 bg-white">
              {selectedRun ? (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-5 bg-emerald-50 rounded-3xl border border-emerald-100">
                      <p className="text-[10px] font-black uppercase text-emerald-600 tracking-widest mb-1">CV Accuracy / R²</p>
                      <p className="text-3xl font-black text-emerald-700">{selectedRun.score ? selectedRun.score.toFixed(4) : 'N/A'}</p>
                    </div>
                    <div className="p-5 bg-indigo-50 rounded-3xl border border-indigo-100">
                      <p className="text-[10px] font-black uppercase text-indigo-600 tracking-widest mb-1">Total Features</p>
                      <p className="text-3xl font-black text-indigo-700">{selectedRun.metrics?.n_features || '-'}</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h4 className="text-xs font-black uppercase text-slate-400 tracking-widest flex items-center gap-2">
                        <Target className="w-4 h-4 text-slate-300" />
                        Hyper-Parameters
                    </h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                      {Object.entries(selectedRun.params || {}).map(([k, v]) => (
                        <div key={k} className="flex justify-between items-center py-2 border-b border-slate-50">
                          <span className="text-sm text-slate-500 font-medium">{k}</span>
                          <span className="text-sm font-mono font-bold bg-slate-100 px-2 py-0.5 rounded text-slate-700">{String(v)}</span>
                        </div>
                      ))}
                      {(!selectedRun.params || Object.keys(selectedRun.params).length === 0) && <p className="text-xs text-slate-400">パラメータが記録されていません</p>}
                    </div>
                  </div>

                  <div className="pt-4 space-y-4">
                    <h4 className="text-xs font-black uppercase text-slate-400 tracking-widest">Training Convergence (Simulation)</h4>
                    <div className="rounded-2xl border border-slate-100 p-2">
                        <Plot
                            data={[
                                { 
                                    y: [0.65, 0.72, 0.78, 0.81, 0.84, selectedRun.score || 0.85], 
                                    type: 'scatter', 
                                    mode: 'lines+markers', 
                                    name: 'CV Score',
                                    line: { color: '#6366f1', width: 3 },
                                    marker: { size: 8, color: '#4338ca' }
                                }
                            ]}
                            layout={{ 
                                margin: { l: 30, r: 10, t: 10, b: 30 }, 
                                xaxis: { gridcolor: '#f1f5f9', zeroline: false }, 
                                yaxis: { gridcolor: '#f1f5f9', zeroline: false }, 
                                height: 160, 
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                autosize: true
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                        />
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-full py-20 text-slate-300">
                  <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6">
                    <FlaskConical className="w-10 h-10 opacity-30" />
                  </div>
                  <p className="font-bold text-slate-400">No Experiment Selected</p>
                  <p className="text-xs text-slate-300 mt-1">左側のテーブルからランを選択してください</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
