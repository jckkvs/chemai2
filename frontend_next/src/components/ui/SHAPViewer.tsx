import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Slider, Tooltip, Tabs, Tab, Box } from '@mui/material';
import { BarChart, Info, Download, RefreshCw, ChevronRight } from 'lucide-react';

// Plotly should be loaded client-side only
const PlotlyChart = dynamic(() => import('react-plotly.js'), { ssr: false });

export interface SHAPData {
  feature_names: string[];
  shap_values: number[][];
  base_value: number;
  predictions: number[];
  feature_values?: number[][];
}

interface SHAPViewerProps {
  jobId: string;
  shapData: SHAPData | null;
  onFeatureSelect?: (feature: string) => void;
}

export const SHAPViewer: React.FC<SHAPViewerProps> = ({ jobId, shapData, onFeatureSelect }) => {
  const [viewMode, setViewMode] = useState<'summary' | 'dependence' | 'force'>('summary');
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [sampleIndex, setSampleIndex] = useState(0);
  const [maxFeatures, setMaxFeatures] = useState(20);

  const featureStats = useMemo(() => {
    if (!shapData) return [];
    return shapData.feature_names.map((name, idx) => {
      const vals = shapData.shap_values.map(row => row[idx]).filter(v => v !== null && !isNaN(v));
      return {
        name, idx,
        mean_abs: vals.length > 0 ? vals.reduce((a, b) => a + Math.abs(b), 0) / vals.length : 0,
        mean: vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
      };
    }).sort((a, b) => b.mean_abs - a.mean_abs);
  }, [shapData]);

  const summaryData = useMemo(() => {
    if (!shapData) return [];
    const top = featureStats.slice(0, maxFeatures).reverse();
    return [{
      type: 'bar', orientation: 'h' as const,
      y: top.map(f => f.name), x: top.map(f => f.mean_abs),
      marker: { color: top.map(f => f.mean >= 0 ? '#22c55e' : '#ef4444') },
      hovertemplate: '<b>%{y}</b><br>|SHAP|: %{x:.4f}<extra></extra>'
    }];
  }, [shapData, featureStats, maxFeatures]);

  if (!shapData) {
    return (
      <div className="p-8 text-center text-slate-400 border border-dashed border-slate-700 rounded-xl">
        <Info className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>SHAPデータが読み込まれていません</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-900/50">
        <div className="flex items-center gap-3">
          <BarChart className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">SHAP 可視化</h3>
          <Tooltip title="SHAP (SHapley Additive exPlanations) はモデル予測の特徴量寄与度を解釈する手法です">
            <Info className="w-4 h-4 text-slate-400 cursor-help" />
          </Tooltip>
        </div>
        <button className="text-slate-400 hover:text-white p-1"><Download className="w-4 h-4" /></button>
      </div>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'rgba(15, 23, 42, 0.3)' }}>
        <Tabs value={viewMode} onChange={(_, v) => setViewMode(v)} variant="scrollable" scrollButtons="auto">
          <Tab label="📊 Summary" value="summary" sx={{ color: '#94a3b8' }} />
          <Tab label="📈 Dependence" value="dependence" sx={{ color: '#94a3b8' }} />
          <Tab label="⚡ Force (Sample)" value="force" sx={{ color: '#94a3b8' }} />
        </Tabs>
      </Box>

      <div className="p-4 space-y-4">
        {viewMode === 'summary' && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">表示特徴量数</span>
              <Slider value={maxFeatures} onChange={(_, v) => setMaxFeatures(v as number)} min={5} max={50} step={5} sx={{ width: 180 }} valueLabelDisplay="auto" />
            </div>
            <div className="h-80">
              <PlotlyChart data={summaryData as any} layout={{
                margin: { t: 20, r: 20, b: 40, l: 120 },
                xaxis: { title: 'Mean |SHAP value|', gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155', automargin: true },
                plot_bgcolor: 'transparent', paper_bgcolor: 'transparent', font: { color: '#94a3b8', size: 11 }
              }} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displayModeBar: false, responsive: true }} />
            </div>
          </>
        )}

        {viewMode === 'dependence' && shapData.feature_values && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <span className="text-sm text-slate-400">特徴量選択:</span>
              <select value={selectedFeature} onChange={(e) => setSelectedFeature(e.target.value)} className="bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500">
                <option value="">選択してください</option>
                {featureStats.slice(0, 30).map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
              </select>
            </div>
            {selectedFeature && (
              <div className="h-72">
                {(() => {
                  const featIdx = shapData.feature_names.indexOf(selectedFeature);
                  if (featIdx < 0) return null;
                  const x = shapData.feature_values.map(r => r[featIdx]);
                  const y = shapData.shap_values.map(r => r[featIdx]);
                  return (
                    <PlotlyChart data={[{
                      type: 'scatter', mode: 'markers', x, y,
                      marker: { size: 6, opacity: 0.7, color: shapData.feature_values.map(r => r[Math.min(featIdx+1, shapData.feature_values![0].length-1)]), colorscale: 'RdYlBu' }
                    }] as any} layout={{
                      xaxis: { title: selectedFeature, gridcolor: '#334155' },
                      yaxis: { title: 'SHAP value', gridcolor: '#334155' },
                      plot_bgcolor: 'transparent', paper_bgcolor: 'transparent', font: { color: '#94a3b8' }
                    }} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displayModeBar: false }} />
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {viewMode === 'force' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">サンプル #{sampleIndex}</span>
              <div className="flex gap-2">
                <button onClick={() => setSampleIndex(Math.max(0, sampleIndex-1))} className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600 transition-colors">←</button>
                <button onClick={() => setSampleIndex(Math.min(shapData.shap_values.length-1, sampleIndex+1))} className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600 transition-colors">→</button>
              </div>
            </div>
            <div className="text-sm text-slate-400">予測値: <span className="text-white font-mono">{shapData.predictions[sampleIndex]?.toFixed(4)}</span> | 基底値: <span className="text-slate-500 font-mono">{shapData.base_value.toFixed(4)}</span></div>
            <div className="space-y-1">
              {(() => {
                const vals = shapData.shap_values[sampleIndex] || [];
                const contributions = shapData.feature_names.map((n, i) => ({ name: n, val: vals[i], abs: Math.abs(vals[i]) })).sort((a,b) => b.abs - a.abs).slice(0, 8);
                return contributions.map(c => (
                  <div key={c.name} className="flex items-center gap-2 text-sm">
                    <span className="w-32 text-slate-300 truncate" title={c.name}>{c.name}</span>
                    <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden relative">
                      <div className={`absolute h-full ${c.val >= 0 ? 'bg-green-500/70' : 'bg-red-500/70'}`} style={{ width: `${Math.min(c.abs * 20, 100)}%`, left: c.val >= 0 ? '50%' : 'auto', right: c.val < 0 ? '50%' : 'auto' }} />
                    </div>
                    <span className={`w-16 text-right font-mono ${c.val >= 0 ? 'text-green-400' : 'text-red-400'}`}>{c.val >= 0 ? '+' : ''}{c.val.toFixed(3)}</span>
                  </div>
                ));
              })()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
