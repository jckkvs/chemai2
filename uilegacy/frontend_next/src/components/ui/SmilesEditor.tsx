import React, { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';

// SSR回避のため Ketcher コンポーネントを動的にインポート
const Ketcher = dynamic(
  () => import('ketcher-react').then(m => m.Ketcher), 
  { 
    ssr: false, 
    loading: () => (
      <div className="flex flex-col items-center justify-center h-[500px] bg-slate-900/50 rounded-b-xl border border-t-0 border-slate-700">
        <Loader2 className="w-10 h-10 animate-spin text-cyan-400 mb-4"/>
        <span className="text-slate-400 font-medium">エディタを初期化中...</span>
      </div>
    ) 
  }
);

interface Props { 
  onChange?: (smiles: string) => void; 
  initial?: string; 
}

export const SmilesEditor: React.FC<Props> = ({ onChange, initial }) => {
  const kRef = useRef<any>(null);
  const [ready, setReady] = useState(false);
  const [sp, setSp] = useState<any>(null);

  useEffect(() => {
    // クライアントサイドでのみ実行
    const { StandaloneStructServiceProvider } = require('ketcher-react');
    setSp(new StandaloneStructServiceProvider());
  }, []);

  useEffect(() => {
    if (initial && ready && kRef.current) {
      kRef.current.setMolecule(initial).catch(() => {});
    }
  }, [initial, ready]);

  if (!sp) return null;

  return (
    <div className="border border-slate-700 rounded-xl overflow-hidden bg-slate-900 shadow-2xl shadow-cyan-900/10">
      <div className="px-5 py-3 bg-slate-800/80 border-b border-slate-700 flex justify-between items-center">
        <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-white tracking-wide">⚗️ 分子構造エディタ</span>
            <span className="text-[10px] bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded uppercase">Standalone</span>
        </div>
        <div className="text-[10px] text-slate-500 font-mono">Ketcher 2.x Engine</div>
      </div>
      <div className="h-[500px] relative ketcher-container">
        <Ketcher 
            staticResourcesUrl="/static" 
            structServiceProvider={sp} 
            onInit={(k: any) => {
                kRef.current = k;
                setReady(true);
                // エディタ内の変更イベントをリッスン
                k.eventBus.on('action', async () => {
                    try {
                        const smiles = await k.generateSmiles();
                        onChange?.(smiles);
                    } catch (e) {
                        // 編集途中の不正な構造などは無視
                    }
                });
            }}
        />
      </div>
      <style jsx global>{`
        .ketcher-container .Ketcher-root {
          background-color: #0f172a !important;
        }
        /* Ketcher のツールバーやボタンのスタイル調整が必要な場合はここに記述 */
      `}</style>
    </div>
  );
};
