import React, { useState } from 'react';
import { FileText, FileSpreadsheet, Code, Download, Loader2, CheckCircle2 } from 'lucide-react';
import { api } from '../../lib/api';

interface ExportPanelProps {
  result: any;
  config: any;
  filename: string;
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ result, config, filename }) => {
  const [status, setStatus] = useState<'idle' | 'exporting' | 'success'>('idle');

  const handleExport = async (fmt: string) => {
    if (!result) return;
    setStatus('exporting');
    try {
      const payload = {
        format: fmt, 
        result, 
        metadata: { filename, config }
      };
      
      const res = await api.post('/api/export/generate', payload, { responseType: 'blob' });
      
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ChemAI_Report_${filename.split('.')[0]}_${new Date().toISOString().split('T')[0]}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setStatus('success');
      setTimeout(() => setStatus('idle'), 3000);
    } catch (e) {
      console.error("Export failed", e);
      alert("エクスポートに失敗しました。サーバーのWeasyPrint依存関係等を確認してください。");
      setStatus('idle');
    }
  };

  if (!result) return null;

  return (
    <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-5 space-y-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2.5">
              <div className="p-1.5 bg-purple-500/20 rounded-lg"><FileText className="w-4 h-4 text-purple-400"/></div>
              結果エクスポート
          </h3>
          {status === 'success' && (
              <span className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-bold bg-emerald-400/10 px-2 py-1 rounded-full animate-bounce">
                  <CheckCircle2 size={12}/> Ready!
              </span>
          )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <button 
            onClick={() => handleExport('json')} 
            disabled={status === 'exporting'}
            className="group flex flex-col items-center gap-2 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50 hover:border-blue-500/50 hover:bg-blue-500/5 transition-all active:scale-95"
        >
            <div className="p-2 bg-blue-500/10 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                <Code className="w-5 h-5 text-blue-400" />
            </div>
            <span className="text-[10px] font-bold text-slate-400 group-hover:text-blue-300 uppercase tracking-widest">JSON</span>
        </button>

        <button 
            onClick={() => handleExport('pdf')} 
            disabled={status === 'exporting'}
            className="group flex flex-col items-center gap-2 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50 hover:border-purple-500/50 hover:bg-purple-500/5 transition-all active:scale-95"
        >
            <div className="p-2 bg-purple-500/10 rounded-lg group-hover:bg-purple-500/20 transition-colors">
                {status === 'exporting' ? <Loader2 className="w-5 h-5 text-purple-400 animate-spin" /> : <Download className="w-5 h-5 text-purple-400" />}
            </div>
            <span className="text-[10px] font-bold text-slate-400 group-hover:text-purple-300 uppercase tracking-widest">PDF Report</span>
        </button>

        <button 
            onClick={() => alert("CSVエクスポートは現在モデル予測データのみ対応しています")} 
            disabled={status === 'exporting'}
            className="group flex flex-col items-center gap-2 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all active:scale-95"
        >
            <div className="p-2 bg-emerald-500/10 rounded-lg group-hover:bg-emerald-500/20 transition-colors">
                <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
            </div>
            <span className="text-[10px] font-bold text-slate-400 group-hover:text-emerald-300 uppercase tracking-widest">CSV</span>
        </button>
      </div>
      
      <p className="text-[10px] text-slate-500 italic text-center px-2">
        ※ PDFレポートには解析サマリー、精度メトリクス、特徴量重要度が含まれます。
      </p>
    </div>
  );
};

const FileSpreadsheet = ({ className }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M8 13h2"/><path d="M8 17h2"/><path d="M14 13h2"/><path d="M14 17h2"/></svg>
);
