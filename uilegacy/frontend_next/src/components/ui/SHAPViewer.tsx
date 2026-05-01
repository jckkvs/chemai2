import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider, Tabs, Tab, Box } from '@mui/material';
import { BarChart3, Download, Info, ChevronLeft, ChevronRight } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SHAPViewerProps { 
  data: { 
    features: string[]; 
    values: number[][]; 
    base: number; 
    predictions: number[]; 
    feature_vals?: number[][]; 
  }; 
}

export const SHAPViewer: React.FC<SHAPViewerProps> = ({ data }) => {
  const [mode, setMode] = useState<'summary'|'force'>('summary');
  const [sampleIdx, setSampleIdx] = useState(0);
  const [maxFeat, setMaxFeat] = useState(20);

  const summaryData = React.useMemo(() => {
    return data.features.map((f, i) => {
      const vals = data.values.map(r => r[i]).filter(v => v !== null && !isNaN(v));
      return { 
        feature: f, 
        mean_abs: vals.reduce((a, b) => a + Math.abs(b), 0) / vals.length, 
        mean: vals.reduce((a, b) => a + b, 0) / vals.length 
      };
    }).sort((a, b) => b.mean_abs - a.mean_abs).slice(0, maxFeat).reverse();
  }, [data, maxFeat]);

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden shadow-xl shadow-black/20">
      <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-900/50">
        <div className="flex items-center gap-3">
            <BarChart3 className="w-5 h-5 text-purple-400"/>
            <h3 className="font-semibold text-white">SHAP 解釈分析</h3>
            <Info className="w-4 h-4 text-slate-500 cursor-help" />
        </div>
        <button className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors">
            <Download className="w-4 h-4"/>
        </button>
      </div>

      <Box sx={{ borderBottom: 1, borderColor: 'rgba(255,255,255,0.1)', bgcolor: 'rgba(15, 23, 42, 0.2)' }}>
        <Tabs 
            value={mode} 
            onChange={(_, v) => setMode(v)} 
            textColor="inherit"
            indicatorColor="secondary"
            sx={{ 
                '& .MuiTab-root': { color: '#94a3b8', fontSize: '0.875rem', textTransform: 'none', minWidth: 100 },
                '& .Mui-selected': { color: '#e879f9' }
            }}
        >
            <Tab label="📊 Summary Plot" value="summary" />
            <Tab label="⚡ Force Plot" value="force" />
        </Tabs>
      </Box>

      <div className="p-6">
        {mode === 'summary' && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex items-center justify-between px-2">
              <span className="text-sm text-slate-400">表示特徴量数</span>
              <div className="flex items-center gap-4">
                  <Slider 
                    value={maxFeat} 
                    onChange={(_, v) => setMaxFeat(v as number)} 
                    min={5} max={50} step={5} 
                    size="small"
                    sx={{ width: 140, color: '#a855f7' }}
                  />
                  <span className="text-xs font-mono text-slate-300 w-6">{maxFeat}</span>
              </div>
            </div>
            <div className="h-[400px]">
                <Plot 
                    data={[{
                        type: 'bar',
                        orientation: 'h',
                        y: summaryData.map(d => d.feature),
                        x: summaryData.map(d => d.mean_abs),
                        marker: { 
                            color: summaryData.map(d => d.mean >= 0 ? '#10b981' : '#f43f5e'),
                            opacity: 0.8
                        },
                        hovertemplate: '<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
                    }]} 
                    layout={{
                        margin: { t: 10, r: 30, b: 40, l: 120 },
                        xaxis: { title: 'Mean |SHAP value| (Average Impact)', gridcolor: '#334155', color: '#94a3b8' },
                        yaxis: { gridcolor: '#334155', color: '#94a3b8', automargin: true },
                        plot_bgcolor: 'transparent',
                        paper_bgcolor: 'transparent',
                        font: { family: 'Inter, sans-serif', size: 11, color: '#94a3b8' },
                        autosize: true
                    }} 
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }} 
                    config={{ displayModeBar: false, responsive: true }}
                />
            </div>
          </div>
        )}

        {mode === 'force' && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex items-center justify-between bg-slate-900/40 p-3 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-4">
                  <span className="text-sm text-slate-400">サンプル表示</span>
                  <div className="flex items-center gap-1">
                      <button onClick={() => setSampleIdx(Math.max(0, sampleIdx - 1))} className="p-1 hover:bg-slate-700 rounded transition-colors text-slate-400"><ChevronLeft size={18}/></button>
                      <span className="text-sm font-mono text-white px-2">#{sampleIdx}</span>
                      <button onClick={() => setSampleIdx(Math.min(data.values.length - 1, sampleIdx + 1))} className="p-1 hover:bg-slate-700 rounded transition-colors text-slate-400"><ChevronRight size={18}/></button>
                  </div>
              </div>
              <div className="text-sm">
                <span className="text-slate-500 mr-2">予測値:</span>
                <span className="text-cyan-400 font-bold font-mono text-base">{data.predictions[sampleIdx]?.toFixed(4)}</span>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Feature Contributions</h4>
              <div className="space-y-2.5 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                {data.features.map((f, i) => {
                  const v = data.values[sampleIdx][i];
                  const absV = Math.abs(v);
                  return (
                    <div key={f} className="group">
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-300 group-hover:text-white transition-colors truncate w-40" title={f}>{f}</span>
                            <span className={`font-mono font-medium ${v >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {v >= 0 ? '+' : ''}{v.toFixed(4)}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="flex-1 h-2 bg-slate-700/50 rounded-full overflow-hidden relative">
                                <div 
                                    className={`absolute h-full rounded-full transition-all duration-500 ${v >= 0 ? 'bg-emerald-500/60 left-1/2' : 'bg-rose-500/60 right-1/2'}`} 
                                    style={{ width: `${Math.min(absV * 15, 50)}%` }}
                                />
                                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-500/50 z-10" />
                            </div>
                        </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
            <div className="pt-4 border-t border-slate-700/50 flex justify-between text-xs text-slate-500 italic">
                <span>Base value: {data.base.toFixed(4)}</span>
                <span>Sum of SHAP values: {data.values[sampleIdx].reduce((a,b)=>a+b,0).toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
