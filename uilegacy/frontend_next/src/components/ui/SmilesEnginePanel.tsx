import React, { useState, useCallback } from 'react';
import { Cpu, Download, AlertCircle, Loader2, CheckCircle2 } from 'lucide-react';
import { api } from '../../lib/api';

interface Engine { key: string; name: string; desc: string; available: boolean; }
const ENGINES: Engine[] = [
  {key:'rdkit', name:'RDKit', desc:'分子量・LogP・TPSA・フィンガープリント', available:true},
  {key:'mordred', name:'Mordred', desc:'1,800+ 2Dトポロジー記述子', available:true},
  {key:'group_contrib', name:'GroupContrib', desc:'基団寄与法（熱物理特性）', available:true},
  {key:'molai', name:'MolAI', desc:'CNN潜在ベクトル + PCA', available:true},
  {key:'skfp', name:'scikit-FP', desc:'ECFP, MACCS, Morgan FP', available:true},
  {key:'uma', name:'UMA', desc:'Universal Molecular Architecture', available:true},
  {key:'mol2vec', name:'Mol2Vec', desc:'Word2Vec分子埋め込み', available:true},
  {key:'padel', name:'PaDEL', desc:'2D/3D記述子 (Java依存)', available:false},
  {key:'molfeat', name:'Molfeat', desc:'統合FPアダプタ', available:true},
  {key:'xtb', name:'XTB', desc:'半経験量子化学 (HOMO/LUMO)', available:false},
  {key:'unipka', name:'UniPKa', desc:'pKa予測・LogD', available:false},
  {key:'cosmo', name:'COSMO-RS', desc:'溶媒和自由エネルギー', available:false},
  {key:'chemprop', name:'Chemprop', desc:'メッセージパッシングNN', available:false},
  {key:'descriptastorus', name:'DescriptaStorus', desc:'高速分子記述子', available:true}
];

export const SmilesEnginePanel: React.FC<{smilesCol:string; filename:string; file:File|null; onCompleted:(cols:string[])=>void}> = ({smilesCol, filename, file, onCompleted}) => {
  const [selected, setSelected] = useState<string[]>(['rdkit']);
  const [status, setStatus] = useState<'idle'|'running'|'completed'|'failed'>('idle');
  const [progress, setProgress] = useState(0);
  const [cachedCols, setCachedCols] = useState<string[]>([]);

  const toggle = (key:string) => {
    if (status === 'running') return;
    setSelected(p=>p.includes(key)?p.filter(k=>k!==key):[...p,key]);
  };

  const runCalculation = useCallback(async () => {
    if (!file || !smilesCol || selected.length === 0) return;
    setStatus('running'); 
    setProgress(10);
    try {
      const fd = new FormData(); 
      fd.append('file', file);
      fd.append('req_json', JSON.stringify({
        smiles_col: smilesCol, 
        engines: selected, 
        options: {}
      }));
      
      const { data } = await api.post('/api/chem/compute', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (p) => setProgress(Math.min(50, (p.loaded / (p.total || 1)) * 50))
      });
      
      setProgress(100);
      setStatus('completed');
      setCachedCols(data.columns || []);
      onCompleted(data.columns || []);
    } catch (err) {
      console.error(err);
      setStatus('failed'); 
    }
  }, [file, smilesCol, selected, onCompleted]);

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6 space-y-6 shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-700/50 pb-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg"><Cpu className="w-5 h-5 text-cyan-400"/></div>
            SMILES 記述子計算エンジン
        </h3>
        {status === 'completed' && (
            <div className="flex items-center gap-2 text-xs bg-emerald-500/10 text-emerald-400 px-3 py-1.5 rounded-full border border-emerald-500/20 animate-fade-in">
                <CheckCircle2 className="w-3.5 h-3.5"/>
                <span>{cachedCols.length} 特徴量を生成しました</span>
            </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {ENGINES.map(e => (
          <label 
            key={e.key} 
            className={`group p-3 rounded-xl border transition-all duration-200 cursor-pointer relative overflow-hidden ${
                selected.includes(e.key) 
                    ? 'border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-500/5' 
                    : e.available 
                        ? 'border-slate-700 bg-slate-900/50 hover:border-slate-500' 
                        : 'border-slate-800 bg-slate-900/20 opacity-40 cursor-not-allowed'
            }`}
          >
            <input 
                type="checkbox" 
                checked={selected.includes(e.key)} 
                onChange={() => toggle(e.key)} 
                disabled={!e.available || status === 'running'} 
                className="hidden"
            />
            <div className="flex flex-col gap-2 relative z-10">
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded-full border flex items-center justify-center transition-colors ${selected.includes(e.key) ? 'border-cyan-400 bg-cyan-400' : 'border-slate-500'}`}>
                    {selected.includes(e.key) && <CheckCircle2 className="w-3 h-3 text-slate-900" />}
                </div>
                <span className="text-sm font-semibold text-slate-200 group-hover:text-white">{e.name}</span>
                {!e.available && <AlertCircle className="w-3 h-3 text-rose-400 ml-auto" title="未インストール"/>}
              </div>
              <div className="text-[10px] text-slate-500 leading-tight group-hover:text-slate-400 transition-colors line-clamp-2">{e.desc}</div>
            </div>
          </label>
        ))}
      </div>

      <div className="pt-2">
        {status === 'running' && (
          <div className="space-y-3 animate-fade-in mb-4">
            <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span>Descriptor generation in progress...</span>
                <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden border border-slate-700">
                <div className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full transition-all duration-700 ease-out shadow-[0_0_8px_rgba(6,182,212,0.5)]" style={{width:`${progress}%`}}/>
            </div>
          </div>
        )}

        <button 
          onClick={runCalculation} 
          disabled={status === 'running' || !selected.length || !smilesCol}
          className={`w-full py-3.5 rounded-xl font-bold text-sm tracking-wide transition-all flex items-center justify-center gap-3 shadow-lg ${
            status === 'running' 
                ? 'bg-slate-700 text-slate-400 cursor-wait' 
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white hover:shadow-cyan-500/20 active:scale-[0.98]'
          }`}
        >
          {status === 'running' ? (
            <><Loader2 className="w-5 h-5 animate-spin"/>計算中...</>
          ) : (
            <><Download className="w-5 h-5"/>記述子計算を開始</>
          )}
        </button>
        
        {!smilesCol && (
            <p className="mt-3 text-center text-xs text-amber-500/80 flex items-center justify-center gap-1.5">
                <AlertCircle size={12}/> SMILES カラムを選択してください
            </p>
        )}
        {status === 'failed' && <p className="mt-3 text-rose-400 text-xs text-center font-medium">計算に失敗しました。詳細についてはサーバーログを確認してください。</p>}
      </div>
    </div>
  );
};
