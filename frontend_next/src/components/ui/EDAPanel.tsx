import React, { useState, useCallback } from 'react';
import { BarChart3, Scatterplot, AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '../../lib/api';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

interface EDAPanelProps { 
  file: File | null; 
  filename: string; 
  targetCol: string; 
  onSelect?: (insight: any) => void; 
}

export const EDAPanel: React.FC<EDAPanelProps> = ({ file, filename, targetCol, onSelect }) => {
  const [tab, setTab] = useState<'metrics'|'corr'|'out'>('metrics');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async (endpoint: string, body: any) => {
    if (!file) return; 
    setLoading(true);
    const fd = new FormData(); 
    fd.append('file', file);
    fd.append('req_json', JSON.stringify(body));
    
    let url = `/api/eda/${endpoint}`;
    if (endpoint === 'corr') {
        // Correct endpoint in router is /correlation
        url = `/api/eda/correlation`;
    }
    const params = endpoint === 'corr' ? { target: targetCol } : {};

    try {
      const response = await api.post(url, fd, { 
        params,
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setData(response.data);
    } catch (e) { 
        console.error(e); 
    } finally { 
        setLoading(false); 
    }
  }, [file, targetCol]);

  React.useEffect(() => { 
    if (file && tab === 'metrics' && !data) {
        fetchData('metrics', { target_col: targetCol, exclude_cols: [] }); 
    }
  }, [file, tab, fetchData, targetCol, data]);

  const handleTabChange = (newTab: 'metrics'|'corr'|'out') => {
    setTab(newTab);
    setData(null);
    if (newTab === 'metrics') fetchData('metrics', { target_col: targetCol, exclude_cols: [] });
    if (newTab === 'corr') fetchData('corr', { method: 'pearson', min_abs: 0.3, top_k: 20 });
    // Outliers endpoint not fully implemented in the provided router snippet but kept for UI
  };

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
      <div className="flex border-b border-slate-700 bg-slate-900/50">
        {[
          { id: 'metrics', label: '📊 統計', icon: BarChart3 },
          { id: 'corr', label: '🔗 相関', icon: Scatterplot },
          { id: 'out', label: '⚠️ 外れ値', icon: AlertTriangle }
        ].map(t => (
          <button 
            key={t.id} 
            onClick={() => handleTabChange(t.id as any)}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${tab === t.id ? 'text-cyan-400 border-b-2 border-cyan-400 bg-slate-800' : 'text-slate-400 hover:text-slate-200'}`}
          >
            <t.icon className="w-4 h-4 inline mr-2 mb-0.5"/>{t.label}
          </button>
        ))}
        <button 
          onClick={() => fetchData(tab, tab === 'metrics' ? { target_col: targetCol, exclude_cols: [] } : { method: 'pearson', min_abs: 0.3, top_k: 20 })} 
          className="px-4 text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}/>
        </button>
      </div>
      
      <div className="p-6">
        {loading && <div className="text-center text-slate-400 py-12 flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <span>分析中...</span>
        </div>}
        
        {!loading && tab === 'metrics' && data?.target_distribution && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in">
            {Object.entries(data.target_distribution).map(([k, v]) => (
              <div key={k} className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{k}</div>
                <div className="text-xl font-mono text-cyan-400">
                    {typeof v === 'number' ? v.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 }) : String(v)}
                </div>
              </div>
            ))}
          </div>
        )}
        
        {!loading && tab === 'corr' && data?.target_corr && (
          <div className="h-64 animate-fade-in">
            <h4 className="text-sm font-medium text-slate-300 mb-4">目的変数との相関（Top 10）</h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={Object.entries(data.target_corr).map(([f, v]) => ({ f, v }))} margin={{ left: 40, right: 20 }}>
                <XAxis type="number" domain={[-1, 1]} stroke="#64748b" fontSize={12} />
                <YAxis type="category" dataKey="f" stroke="#64748b" width={80} fontSize={10} />
                <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#22d3ee' }}
                />
                <Bar dataKey="v">
                  {Object.entries(data.target_corr).map(([, v], i) => (
                    <Cell key={i} fill={(v as number) >= 0 ? '#10b981' : '#ef4444'} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {!loading && tab === 'out' && (
          <div className="text-center py-8 text-slate-500 italic">
            外れ値検出エンジン準備中...
          </div>
        )}

        {!loading && !data && !loading && (
          <div className="text-center py-12 text-slate-500">
            {file ? "データを読み込んでください" : "ファイルが選択されていません"}
          </div>
        )}
      </div>
    </div>
  );
};
