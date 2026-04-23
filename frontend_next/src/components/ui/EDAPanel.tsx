import React, { useState, useCallback } from 'react';
import { BarChart3, Scatterplot, AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '../../lib/api';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

interface EDAPanelProps {
  file: File | null;
  filename: string;
  targetCol: string;
  onInsightSelect?: (insight: any) => void;
}

type EDATab = 'metrics' | 'correlation' | 'outliers';

export const EDAPanel: React.FC<EDAPanelProps> = ({ file, filename, targetCol, onInsightSelect }) => {
  const [activeTab, setActiveTab] = useState<EDATab>('metrics');
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [correlation, setCorrelation] = useState<any>(null);
  const [outliers, setOutliers] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('request_json', JSON.stringify({ target_col: targetCol, exclude_cols: [] }));
      const { data } = await api.post('/api/eda/metrics', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setMetrics(data);
    } catch (err: any) { setError(err.response?.data?.detail || '統計量計算失敗'); }
    finally { setLoading(false); }
  }, [file, targetCol]);

  const fetchCorrelation = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('request_json', JSON.stringify({ method: 'pearson', min_abs_corr: 0.3, top_k: 20 }));
      formData.append('target_col', targetCol);
      const { data } = await api.post('/api/eda/correlation', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setCorrelation(data);
    } catch (err: any) { setError(err.response?.data?.detail || '相関分析失敗'); }
    finally { setLoading(false); }
  }, [file, targetCol]);

  const fetchOutliers = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', 'iqr'); formData.append('threshold', '1.5');
      const { data } = await api.post('/api/eda/outliers', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setOutliers(data);
    } catch (err: any) { setError(err.response?.data?.detail || '外れ値検出失敗'); }
    finally { setLoading(false); }
  }, [file]);

  React.useEffect(() => { if (file && activeTab === 'metrics' && !metrics) fetchMetrics(); }, [file, activeTab, fetchMetrics]);

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
      <div className="flex border-b border-slate-700 bg-slate-900/50">
        {[
          { id: 'metrics', label: '📊 基本統計', icon: BarChart3 },
          { id: 'correlation', label: '🔗 相関分析', icon: Scatterplot },
          { id: 'outliers', label: '⚠️ 外れ値', icon: AlertTriangle },
        ].map((tab) => (
          <button key={tab.id} onClick={() => {
            setActiveTab(tab.id as EDATab);
            if (tab.id === 'correlation' && !correlation) fetchCorrelation();
            if (tab.id === 'outliers' && !outliers) fetchOutliers();
          }} className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors ${activeTab === tab.id ? 'text-cyan-400 border-b-2 border-cyan-400 bg-slate-800' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'}`}>
            <tab.icon className="w-4 h-4" /> {tab.label}
          </button>
        ))}
        <button onClick={() => { setMetrics(null); setCorrelation(null); setOutliers(null); if (activeTab === 'metrics') fetchMetrics(); if (activeTab === 'correlation') fetchCorrelation(); if (activeTab === 'outliers') fetchOutliers(); }} className="px-4 text-slate-400 hover:text-cyan-400 transition-colors" title="再計算">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="p-6">
        {loading && <div className="flex items-center justify-center py-12"><div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin" /><span className="ml-3 text-slate-400">分析中...</span></div>}
        {error && <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">{error}</div>}

        {activeTab === 'metrics' && metrics && !loading && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="行数" value={metrics.metrics?.shape?.rows?.toLocaleString()} />
              <StatCard label="列数" value={metrics.metrics?.shape?.columns?.toLocaleString()} />
              <StatCard label="数値列" value={metrics.numeric_columns?.length || 0} />
              <StatCard label="カテゴリ列" value={metrics.categorical_columns?.length || 0} />
            </div>
            {metrics.metrics?.target_distribution && (
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                <h4 className="font-semibold text-white mb-3">目的変数「{targetCol}」の分布</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  {Object.entries(metrics.metrics.target_distribution).map(([k, v]) => (
                    <div key={k} className="flex justify-between"><span className="text-slate-400 capitalize">{k}:</span><span className="text-white font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span></div>
                  ))}
                </div>
              </div>
            )}
            <div>
              <h4 className="font-semibold text-white mb-3">欠損値サマリー</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {Object.entries(metrics.metrics.missing_rate || {}).filter(([, rate]: any) => rate > 0).sort(([, a]: any, [, b]: any) => (b as number) - (a as number)).slice(0, 10).map(([col, rate]: any) => (
                  <div key={col} className="flex items-center gap-3 text-sm">
                    <span className="flex-1 text-slate-300 truncate" title={col}>{col}</span>
                    <div className="w-24 bg-slate-700 rounded-full h-2"><div className={`h-2 rounded-full ${rate > 30 ? 'bg-red-500' : rate > 10 ? 'bg-yellow-500' : 'bg-green-500'}`} style={{ width: `${Math.min(rate, 100)}%` }} /></div>
                    <span className="text-slate-400 w-12 text-right">{rate}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'correlation' && correlation && !loading && (
          <div className="space-y-6">
            {correlation.target_correlation && (
              <div>
                <h4 className="font-semibold text-white mb-3">目的変数との相関（上位10）</h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(correlation.target_correlation).map(([f, v]: any) => ({ feature: f, corr: v }))} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
                      <XAxis type="number" domain={[-1, 1]} stroke="#64748b" />
                      <YAxis type="category" dataKey="feature" stroke="#64748b" width={50} />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                      <Bar dataKey="corr">{Object.entries(correlation.target_correlation).map(([, v]: any, i: number) => (<Cell key={i} fill={(v as number) >= 0 ? '#22c55e' : '#ef4444'} />))}</Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            {correlation.top_pairs?.length > 0 && (
              <div>
                <h4 className="font-semibold text-white mb-3">特徴量間の強い相関ペア</h4>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {correlation.top_pairs.map((pair: any, i: number) => (
                    <button key={i} onClick={() => onInsightSelect?.({ type: 'correlation', ...pair })} className="w-full flex items-center justify-between p-2 text-sm bg-slate-900/50 hover:bg-slate-800 rounded border border-slate-700 hover:border-cyan-500/50 transition-colors text-left">
                      <span className="text-slate-300 truncate flex-1">{pair.feature1} ↔ {pair.feature2}</span>
                      <span className={`font-bold ${pair.correlation >= 0 ? 'text-green-400' : 'text-red-400'}`}>{pair.correlation >= 0 ? '+' : ''}{pair.correlation.toFixed(3)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'outliers' && outliers && !loading && (
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm text-slate-400">
              <span>検出方法: {outliers.method.toUpperCase()}</span>
              <span>閾値: {outliers.threshold}</span>
            </div>
            <div className="space-y-3 max-h-60 overflow-y-auto">
              {Object.entries(outliers.results || {}).map(([col, info]: any) => (
                <div key={col} className="p-3 bg-slate-900/50 rounded border border-slate-700">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white truncate flex-1" title={col}>{col}</span>
                    <span className={`text-sm font-bold ${(info.outlier_rate || 0) > 10 ? 'text-red-400' : (info.outlier_rate || 0) > 5 ? 'text-yellow-400' : 'text-green-400'}`}>{(info.outlier_rate || 0).toFixed(1)}%</span>
                  </div>
                  <div className="text-xs text-slate-400 space-y-1">
                    <div>外れ値数: {info.outlier_count?.toLocaleString() || 'N/A'}</div>
                    {info.bounds && <div>閾値: [{info.bounds.lower?.toFixed(3)}, {info.bounds.upper?.toFixed(3)}]</div>}
                  </div>
                  {(info.outlier_rate || 0) > 5 && <button onClick={() => onInsightSelect?.({ type: 'outlier', column: col, ...info })} className="mt-2 text-xs text-cyan-400 hover:text-cyan-300 underline">対応を検討 →</button>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const StatCard: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <div className="bg-slate-900/50 rounded-lg p-4 text-center border border-slate-700">
    <div className="text-2xl font-bold text-cyan-400">{value}</div>
    <div className="text-xs text-slate-400 mt-1">{label}</div>
  </div>
);
