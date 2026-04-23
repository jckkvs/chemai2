import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';

// Next.js SSR 対策: Ketcher はブラウザ依存のため dynamic import 必須
const KetcherStandalone = dynamic(
  () => import('./KetcherWrapper').then(mod => mod.KetcherWrapper),
  { ssr: false, loading: () => (
    <div className="flex items-center justify-center h-[500px] bg-slate-900/50">
      <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
      <span className="ml-3 text-slate-400">エディタを読み込み中...</span>
    </div>
  ) }
);

interface SmilesEditorProps {
  onSmilesChange: (smiles: string) => void;
  initialSmiles?: string;
  className?: string;
}

export const SmilesEditor: React.FC<SmilesEditorProps> = ({ onSmilesChange, initialSmiles, className }) => {
  return (
    <div className={`rounded-xl border border-slate-700 overflow-hidden bg-slate-900 ${className}`}>
      <div className="px-4 py-2 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-300">⚗️ 分子構造エディタ</span>
        </div>
        <span className="text-xs text-slate-500 font-mono">Ketcher 2.x</span>
      </div>
      <div className="h-[500px]">
        <KetcherStandalone onMolExport={onSmilesChange} initialSmiles={initialSmiles} />
      </div>
    </div>
  );
};
