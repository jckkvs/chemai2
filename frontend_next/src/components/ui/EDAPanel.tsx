import React, { useState, useCallback } from 'react';
import { BarChart3, Scatterplot, AlertTriangle, RefreshCw, Layers } from 'lucide-react';
import { api } from '../../lib/api';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ScatterChart, Scatter } from 'recharts';

interface EDAPanelProps {
  file: File | null;
  filename: string;
  targetCol: string;
  onInsightSelect?: (insight: any) => void;
}

type EDATab = 'metrics' | 'correlation' | 'outliers' | 'dim_red';

export const EDAPanel: React.FC<EDAPanelProps> = ({ file, filename, targetCol, onInsightSelect }) => {
  const [activeTab, setActiveTab] = useState<EDATab>('metrics');
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [correlation, setCorrelation] = useState<any>(null);
  const [outliers, setOutliers] = useState<any>(null);
  const [dimRed, setDimRed] = useState<any>(null);
  const [dimMethod, setDimMethod] = useState<'pca'|'tsne'>('pca');
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async (tab: EDATab) => {
    if (!file) return;
    setLoading(true); 
    setError(null);
    try {
      const fd = new FormData(); 
      fd.append('file', file);
      
      if (tab === 'metrics') {
        fd.append('req_json', JSON.stringify({ target_col: targetCol, exclude_cols: [] }));
        const { data } = await api.post('/api/eda/metrics', fd, { headers: {'Content-Type':'multipart/form-data'} });
        setMetrics(data);
      } else if (tab === 'correlation') {
        fd.append('req_json', JSON.stringify({ method: 'pearson', min_abs: 0.3, top_k: 20 }));
        const { data } = await api.post('/api/eda/correlation', fd, { 
            params: { target_col: targetCol },
            headers: {'Content-Type':'multipart/form-data'} 
        });
        setCorrelation(data);
      } else if (tab === 'outliers') {
        fd.append('method', 'iqr'); 
        fd.append('threshold', '1.5');
        const { data } = await api.post('/api/eda/outliers', fd, { headers: {'Content-Type':'multipart/form-data'} });
        setOutliers(data);
      } else if (tab === 'dim_red') {
        const { data } = await api.post('/api/eda/dim_reduction', fd, { 
            params: { 
                method: dimMethod,
                n_components: 2,
                perplexity: 30,
                target_col: targetCol
            },
            headers: {'Content-Type':'multipart/form-data'} 
        });
        setDimRed(data);
      }
    } catch (err: any) { 
        setError(err.response?.data?.detail || '分析に失敗しました'); 
    } finally { 
        setLoading(false); 
    }
  }, [file, targetCol, dimMethod]);

  React.useEffect(() => { 
    if (file && activeTab === 'metrics' && !metrics) fetchData('metrics'); 
  }, [file, activeTab, fetchData, metrics]);

  const handleTabChange = (t: EDATab) => {
    setActiveTab(t);
    if (t === 'metrics' && !metrics) fetchData('metrics');
    if (t === 'correlation' && !correlation) fetchData('correlation');
    if (t === 'outliers' && !outliers) fetchData('outliers');
    if (t === 'dim_red' && !dimRed) fetchData('dim_red');
  };

  const tabs: {id: EDATab; label: string; icon: React.ElementType}[] = [
    { id: 'metrics', label: '📊 基本統計', icon: BarChart3 },
    { id: 'correlation', label: '🔗 相関', icon: Scatterplot },
    { id: 'outliers', label: '⚠️ 外れ値', icon: AlertTriangle },
    { id: 'dim_red', label: '📉 次元削減', icon: Layers }
  ];

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
      <div className="flex border-b border-slate-700 bg-slate-900/50">
        {tabs.map(t => (
          <button key={t.id} onClick={() => handleTabChange(t.id)}
            className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-all ${activeTab===t.id ? 'text-cyan-400 border-b-2 border-cyan-400 bg-slate-800' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'}`}>
            <t.icon className="w-4 h-4" /> {t.label}
          </button>
        ))}
        <button onClick={() => fetchData(activeTab)} className="px-4 text-slate-400 hover:text-cyan-400 transition-colors">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="p-6">
        {loading && <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-4">
            <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <span>分析中...</span>
        </div>}
        
        {error && <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm mb-4">{error}</div>}

        {!loading && activeTab === 'metrics' && metrics && <MetricsTab data={metrics} target={targetCol} />}
        {!loading && activeTab === 'correlation' && correlation && <CorrelationTab data={correlation} target={targetCol} />}
        {!loading && activeTab === 'outliers' && outliers && <OutliersTab data={outliers} />}
        {!loading && activeTab === 'dim_red' && (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-center gap-4 bg-slate-900/50 p-3 rounded-lg border border-slate-700">
              <span className="text-sm text-slate-400">手法:</span>
              <select value={dimMethod} onChange={e => setDimMethod(e.target.value as any)} className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-white focus:ring-1 focus:ring-cyan-500 outline-none">
                <option value="pca">PCA (主成分分析)</option>
                <option value="tsne">t-SNE (埋め込み表示)</option>
              </select>
              <button onClick={()=>fetchData('dim_red')} className="ml-auto px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm transition-colors">再計算</button>
            </div>
            {dimRed && <DimRedTab data={dimRed} />}
          </div>
        )}
      </div>
    </div>
  );
};

const MetricsTab: React.FC<{data:any; target:string}> = ({data, target}) => (
  <div className="space-y-6 animate-fade-in">
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <StatCard label="行数" value={data.metrics?.shape?.rows?.toLocaleString()} />
      <StatCard label="列数" value={data.metrics?.shape?.columns?.toLocaleString()} />
      <StatCard label="数値列" value={data.numeric_columns?.length || 0} />
      <StatCard label="カテゴリ列" value={data.categorical_columns?.length || 0} />
    </div>
    
    {data.metrics?.target_distribution && (
      <div className="bg-slate-900/50 rounded-lg p-5 border border-slate-700">
        <h4 className="text-sm font-semibold text-cyan-400 mb-4 uppercase tracking-wider">目的変数「{target}」の分布統計</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
          {Object.entries(data.metrics.target_distribution).map(([k,v])=>(
            <div key={k} className="flex flex-col"><span className="text-slate-500 text-xs uppercase mb-1">{k}</span><span className="text-white font-mono text-lg">{typeof v==='number'?v.toFixed(4):v}</span></div>
          ))}
        </div>
      </div>
    )}
    
    <div>
      <h4 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">欠損値サマリー（Top 10）</h4>
      <div className="space-y-3 bg-slate-900/30 p-4 rounded-lg border border-slate-700/50 max-h-60 overflow-y-auto">
        {Object.entries(data.metrics.missing_rate||{}).filter(([,r]:any)=>r>0).sort(([,a]:any,[,b]:any)=>b-a).slice(0,10).map(([c,r]:any)=>(
          <div key={c} className="flex items-center gap-4 text-sm group">
            <span className="flex-1 text-slate-300 truncate font-medium group-hover:text-white transition-colors">{c}</span>
            <div className="w-40 bg-slate-800 rounded-full h-1.5 overflow-hidden"><div className={`h-full rounded-full transition-all duration-1000 ${r>30?'bg-rose-500':r>10?'bg-amber-500':'bg-emerald-500'}`} style={{width:`${Math.min(r,100)}%`}}/></div>
            <span className="text-slate-400 w-12 text-right font-mono">{r}%</span>
          </div>
        ))}
        {Object.values(data.metrics.missing_rate||{}).every((r:any)=>r===0) && <div className="text-center text-slate-500 py-4 italic">欠損値は見つかりませんでした</div>}
      </div>
    </div>
  </div>
);

const CorrelationTab: React.FC<{data:any; target:string}> = ({data, target}) => (
  <div className="space-y-8 animate-fade-in">
    {data.target_correlation && (
      <div>
        <h4 className="text-sm font-semibold text-cyan-400 mb-4 uppercase tracking-wider">目的変数との相関（上位10）</h4>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(data.target_correlation).map(([f,v]:any)=>({f,v}))} layout="vertical" margin={{top:5,right:30,left:40,bottom:5}}>
              <XAxis type="number" domain={[-1,1]} stroke="#475569" fontSize={11}/><YAxis type="category" dataKey="f" stroke="#475569" width={80} fontSize={10}/>
              <Tooltip cursor={{fill:'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor:'#1e293b',border:'1px solid #334155',borderRadius:'8px'}}/>
              <Bar dataKey="v" radius={[0,4,4,0]}>{Object.entries(data.target_correlation).map(([,v]:any,i:number)=><Cell key={i} fill={(v as number)>=0?'#10b981':'#f43f5e'} fillOpacity={0.7}/>)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    )}
    
    <div>
      <h4 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">特徴量間の強相関ペア</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
        {data.top_pairs?.map((p:any,i:number)=>(
          <div key={i} className="flex items-center justify-between p-3 text-xs bg-slate-900/50 rounded-lg border border-slate-700/50 hover:border-slate-500 transition-colors">
            <div className="flex flex-col gap-1 flex-1 overflow-hidden">
                <span className="text-slate-300 truncate">{p.feature1}</span>
                <span className="text-slate-500">vs</span>
                <span className="text-slate-300 truncate">{p.feature2}</span>
            </div>
            <div className={`text-sm font-bold font-mono ml-4 px-2 py-1 rounded ${p.correlation>=0?'text-emerald-400 bg-emerald-400/10':'text-rose-400 bg-rose-400/10'}`}>{p.correlation>=0?'+':''}{p.correlation.toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const OutliersTab: React.FC<{data:any}> = ({data}) => (
  <div className="space-y-4 animate-fade-in">
    <div className="flex items-center justify-between bg-slate-900/50 p-3 rounded-lg border border-slate-700 text-xs text-slate-500">
      <div className="flex gap-4"><span>手法: <span className="text-white uppercase">{data.method}</span></span><span>閾値: <span className="text-white">{data.threshold}</span></span></div>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
        {Object.entries(data.results||{}).sort(([,a]:any,[,b]:any)=>b.outlier_rate - a.outlier_rate).map(([c,info]:any)=>(
          <div key={c} className="p-4 bg-slate-900/30 rounded-lg border border-slate-700/50 group hover:border-amber-500/50 transition-colors">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium text-slate-200 truncate flex-1 group-hover:text-white">{c}</span>
              <span className={`text-sm font-bold font-mono px-2 py-0.5 rounded ${(info.outlier_rate||0)>10?'bg-rose-500/20 text-rose-400':(info.outlier_rate||0)>5?'bg-amber-500/20 text-amber-400':'bg-emerald-500/20 text-emerald-400'}`}>{(info.outlier_rate||0).toFixed(1)}%</span>
            </div>
            <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-[10px] text-slate-500 uppercase tracking-tighter"><span>外れ値数: {info.outlier_count?.toLocaleString()}</span><span>範囲: [{info.bounds.lower?.toFixed(2)}, {info.bounds.upper?.toFixed(2)}]</span></div>
                <div className="w-full bg-slate-800 rounded-full h-1"><div className={`h-full rounded-full ${info.outlier_rate > 10 ? 'bg-rose-500' : 'bg-amber-500'}`} style={{width: `${Math.min(info.outlier_rate, 100)}%`}} /></div>
            </div>
          </div>
        ))}
    </div>
  </div>
);

const DimRedTab: React.FC<{data:any}> = ({data}) => (
  <div className="space-y-4 animate-fade-in">
    <div className="flex items-center gap-4 text-xs text-slate-500 font-mono">
      <span className="bg-slate-900 px-2 py-1 rounded">サンプル数: {data.n_samples}</span>
      {data.explained_variance?.length > 0 && <span className="bg-slate-900 px-2 py-1 rounded">第1主成分 寄与率: {(data.explained_variance[0]*100).toFixed(1)}%</span>}
    </div>
    <div className="h-96 bg-slate-900/80 rounded-xl border border-slate-700 p-4 shadow-inner relative overflow-hidden">
      <div className="absolute inset-0 opacity-10 pointer-events-none" style={{backgroundImage:'radial-gradient(#475569 1px, transparent 1px)',backgroundSize:'20px 20px'}}></div>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{top:20,right:30,bottom:40,left:40}}>
          <XAxis type="number" dataKey="comp_0" name="Component 1" stroke="#475569" label={{value:'Component 1',position:'insideBottomRight',offset:-10, fill:'#64748b', fontSize:10}} fontSize={10} tick={{fill:'#64748b'}}/>
          <YAxis type="number" dataKey="comp_1" name="Component 2" stroke="#475569" label={{value:'Component 2',angle:-90,position:'insideLeft', fill:'#64748b', fontSize:10}} fontSize={10} tick={{fill:'#64748b'}}/>
          <Tooltip 
            cursor={{strokeDasharray:'3 3', stroke:'#94a3b8'}} 
            contentStyle={{backgroundColor:'rgba(15, 23, 42, 0.9)',border:'1px solid #334155', borderRadius:'12px', backdropFilter:'blur(4px)'}}
            itemStyle={{fontSize:'11px', color:'#22d3ee'}}
          />
          <Scatter name="Data Points" data={data.embeddings} fill="#06b6d4" fillOpacity={0.6}>
            {data.embeddings.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={entry.target !== undefined ? (typeof entry.target === 'number' ? `hsl(${200 + entry.target * 10}, 70%, 60%)` : '#06b6d4') : '#06b6d4'} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  </div>
);

const StatCard: React.FC<{label:string;value:string|number}> = ({label,value}) => (
  <div className="bg-slate-900/50 rounded-xl p-5 text-center border border-slate-700/50 hover:bg-slate-900 hover:border-slate-500 transition-all cursor-default">
    <div className="text-3xl font-bold text-cyan-400 font-mono tracking-tight">{value}</div>
    <div className="text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-semibold">{label}</div>
  </div>
);
