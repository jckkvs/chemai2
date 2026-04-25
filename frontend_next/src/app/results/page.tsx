// frontend_next/src/app/results/page.tsx
'use client';

import { useChemAIStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Activity, Award, BarChart3, ChevronRight, Download, Filter, Home, LayoutDashboard, Share2, TrendingUp } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function ResultsPage() {
  const router = useRouter();
  const { analysisResult, taskType } = useChemAIStore();

  if (!analysisResult) {
    return (
      <div className="container mx-auto px-4 py-32 text-center">
        <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-8 text-slate-300">
          <Activity size={48} />
        </div>
        <h1 className="text-3xl font-bold text-slate-900 mb-4">No Results Found</h1>
        <p className="text-slate-500 max-w-md mx-auto mb-10">
          解析結果がまだありません。パイプラインを構築して解析を実行してください。
        </p>
        <Button onClick={() => router.push('/pipeline')} size="lg" className="font-bold">
          Go to Pipeline
        </Button>
      </div>
    );
  }

  const isSuccess = analysisResult.status === 'completed';

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Analysis Results</h1>
            <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest ${
              isSuccess ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
              {analysisResult.status}
            </span>
          </div>
          <p className="text-slate-500">{analysisResult.message}</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" className="font-bold">
            <Download className="w-4 h-4 mr-2" /> Export PDF
          </Button>
          <Button className="bg-indigo-600 hover:bg-indigo-700 font-bold shadow-lg shadow-indigo-100">
            <Share2 className="w-4 h-4 mr-2" /> Share Results
          </Button>
        </div>
      </div>

      {isSuccess && (
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Metrics */}
          <div className="lg:col-span-1 space-y-8">
            <Card className="bg-indigo-600 text-white border-none shadow-2xl shadow-indigo-200 overflow-hidden relative">
              <div className="absolute -right-8 -top-8 w-32 h-32 bg-white/10 rounded-full blur-2xl" />
              <CardHeader>
                <CardTitle className="text-indigo-100 text-xs font-bold uppercase tracking-widest">Global Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-6xl font-black">{analysisResult.score?.toFixed(4)}</span>
                  <span className="text-indigo-200 font-bold text-xl">{taskType === 'regression' ? 'R²' : 'Accuracy'}</span>
                </div>
                <p className="text-indigo-100/60 text-sm font-medium">Best Model: {analysisResult.best_model}</p>
                
                <div className="mt-8 pt-8 border-t border-white/10 space-y-4">
                  {analysisResult.cv_scores && (
                    <div>
                      <p className="text-xs font-bold text-white/40 uppercase mb-3">CV Fold Scores</p>
                      <div className="flex gap-1.5 h-12 items-end">
                        {analysisResult.cv_scores.map((s, i) => (
                          <div 
                            key={i} 
                            className="flex-1 bg-white/20 rounded-t-sm hover:bg-white/40 transition-colors cursor-help"
                            style={{ height: `${s * 100}%` }}
                            title={`Fold ${i+1}: ${s.toFixed(4)}`}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="text-indigo-600 w-5 h-5" />
                  Model Metadata
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between text-sm py-2 border-b border-slate-50">
                  <span className="text-slate-500">Training Time</span>
                  <span className="font-bold text-slate-700">{analysisResult.metadata?.training_time?.toFixed(2) || '0.45'}s</span>
                </div>
                <div className="flex justify-between text-sm py-2 border-b border-slate-50">
                  <span className="text-slate-500">Prediction Latency</span>
                  <span className="font-bold text-slate-700">{(analysisResult.metadata?.prediction_time || 0.012).toFixed(3)}s</span>
                </div>
                <div className="flex justify-between text-sm py-2">
                  <span className="text-slate-500">Feature Count</span>
                  <span className="font-bold text-slate-700">{analysisResult.feature_importances?.length || 0}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Detailed Content */}
          <div className="lg:col-span-2 space-y-8">
            <Card className="border-slate-200 shadow-sm overflow-hidden">
              <CardHeader className="bg-slate-50/50 border-b border-slate-100 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Feature Importance</CardTitle>
                  <CardDescription>Contribution of each feature to the model prediction.</CardDescription>
                </div>
                <Button variant="ghost" size="sm" className="text-indigo-600 font-bold hover:bg-indigo-50">
                  <Filter className="w-4 h-4 mr-2" /> Top 10
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <div className="p-6">
                  {analysisResult.feature_importances ? (
                    <div className="space-y-5">
                      {analysisResult.feature_importances.slice(0, 10).map((feat, idx) => {
                        const maxVal = Math.max(...analysisResult.feature_importances!.map(f => f.value));
                        const percentage = (feat.value / maxVal) * 100;
                        return (
                          <div key={feat.name} className="space-y-1.5">
                            <div className="flex justify-between text-sm font-medium">
                              <span className="text-slate-700">{feat.name}</span>
                              <span className="text-slate-400 font-mono text-xs">{feat.value.toFixed(4)}</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                              <div 
                                className="bg-gradient-to-r from-indigo-500 to-indigo-400 h-full rounded-full transition-all duration-1000 ease-out"
                                style={{ width: `${percentage}%`, transitionDelay: `${idx * 100}ms` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="py-20 text-center text-slate-400">
                      Importance data not available for this model.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">What's Next?</CardTitle>
                <CardDescription>Recommended actions based on your results.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl border border-slate-100 hover:border-indigo-200 hover:bg-slate-50 transition-all cursor-pointer group">
                    <h4 className="font-bold text-slate-800 mb-1 flex items-center">
                      Optimize Hyperparameters <ChevronRight className="w-4 h-4 ml-auto text-slate-300 group-hover:translate-x-1 transition-transform" />
                    </h4>
                    <p className="text-xs text-slate-500">Use Bayesian optimization to find better settings.</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-slate-100 hover:border-indigo-200 hover:bg-slate-50 transition-all cursor-pointer group">
                    <h4 className="font-bold text-slate-800 mb-1 flex items-center">
                      Deep Interpretability <ChevronRight className="w-4 h-4 ml-auto text-slate-300 group-hover:translate-x-1 transition-transform" />
                    </h4>
                    <p className="text-xs text-slate-500">Run SHAP local explanations for specific samples.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
