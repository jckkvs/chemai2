import React, { useState, useEffect, useCallback } from 'react';
import { Bookmark, History, Save, Trash2, Download, RotateCcw, Plus, Info } from 'lucide-react';
import { api } from '../../lib/api';

interface PresetManagerProps {
  currentConfig: any;
  onLoad: (config: any) => void;
  onExport?: () => void;
}

export const PresetManager: React.FC<PresetManagerProps> = ({ currentConfig, onLoad, onExport }) => {
  const [presets, setPresets] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [tab, setTab] = useState<'presets'|'history'>('presets');
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
        const [pRes, hRes] = await Promise.all([
            api.get('/api/presets/list'),
            api.get('/api/presets/history')
        ]);
        setPresets(pRes.data);
        setHistory([...hRes.data].reverse());
    } catch (e) {
        console.error("Failed to fetch presets/history", e);
    } finally {
        setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSave = async () => {
    if (!name.trim()) return;
    try {
        await api.post('/api/presets/save', { name, description: desc, config: currentConfig });
        setName(''); 
        setDesc(''); 
        fetchData();
    } catch (e) {
        alert("プリセットの保存に失敗しました");
    }
  };

  const handleLoad = (p: any) => { 
    onLoad(p.config); 
  };

  const handleDelete = async (n: string) => { 
    if (!confirm(`プリセット「${n}」を削除しますか？`)) return;
    await api.delete(`/api/presets/${n}`); 
    fetchData(); 
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-5 space-y-5 shadow-2xl backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-slate-700/50 pb-3">
        <div className="flex gap-2 p-1 bg-slate-900/50 rounded-xl">
            <button 
                onClick={() => setTab('presets')} 
                className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-lg transition-all ${tab === 'presets' ? 'bg-cyan-500 text-slate-900 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <Bookmark size={14}/> プリセット
            </button>
            <button 
                onClick={() => setTab('history')} 
                className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-lg transition-all ${tab === 'history' ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/20' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <History size={14}/> 履歴
            </button>
        </div>
        <button onClick={onExport} className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors" title="エクスポート">
            <Download size={18}/>
        </button>
      </div>

      {tab === 'presets' && (
        <div className="space-y-4 animate-fade-in">
          <div className="space-y-3 bg-slate-900/40 p-4 rounded-xl border border-slate-700/50">
            <div className="flex gap-2">
                <input 
                    value={name} 
                    onChange={e => setName(e.target.value)} 
                    placeholder="プリセット名..." 
                    className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-cyan-500/50 outline-none transition-all placeholder:text-slate-600"
                />
                <button 
                    onClick={handleSave} 
                    disabled={!name.trim()}
                    className="px-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:opacity-50 text-white rounded-lg transition-all flex items-center gap-2 font-bold text-xs"
                >
                    <Plus size={16}/> 保存
                </button>
            </div>
            <textarea 
                value={desc} 
                onChange={e => setDesc(e.target.value)} 
                placeholder="説明文（任意）" 
                rows={1}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300 focus:ring-2 focus:ring-cyan-500/50 outline-none transition-all placeholder:text-slate-600 resize-none"
            />
          </div>

          <div className="space-y-2 max-h-56 overflow-y-auto pr-2 custom-scrollbar">
            {presets.length === 0 && !loading && (
                <div className="text-center py-10 text-slate-600 flex flex-col items-center gap-2">
                    <Info size={24} className="opacity-20"/>
                    <span className="text-xs italic">プリセットが登録されていません</span>
                </div>
            )}
            {presets.map(p => (
              <div key={p.name} className="group flex items-center justify-between p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/50 hover:border-cyan-500/50 transition-all hover:translate-x-1">
                <div className="flex flex-col gap-1 overflow-hidden">
                    <div className="text-sm font-bold text-slate-200 group-hover:text-white transition-colors">{p.name}</div>
                    <div className="text-[10px] text-slate-500 truncate">{p.desc || '説明なし'}</div>
                </div>
                <div className="flex gap-1.5 ml-4">
                  <button onClick={() => handleLoad(p)} className="p-2 text-cyan-400 hover:bg-cyan-400/10 rounded-lg transition-colors" title="適用"><RotateCcw size={16}/></button>
                  <button onClick={() => handleDelete(p.name)} className="p-2 text-rose-400 hover:bg-rose-400/10 rounded-lg transition-colors" title="削除"><Trash2 size={16}/></button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'history' && (
        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar animate-fade-in">
          {history.length === 0 && !loading && (
             <div className="text-center py-12 text-slate-600 flex flex-col items-center gap-2">
                 <History size={24} className="opacity-20"/>
                 <span className="text-xs italic">解析履歴がありません</span>
             </div>
          )}
          {history.map((h, i) => (
            <div key={i} className="flex flex-col gap-2 p-4 bg-slate-900/40 rounded-xl border border-slate-700/30 hover:border-purple-500/30 transition-colors">
              <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono uppercase tracking-wider">
                <span>{h.timestamp?.split('T')[0]} {h.timestamp?.split('T')[1]?.substring(0,5)}</span>
                <span className="bg-slate-800 px-2 py-0.5 rounded text-slate-400">ID: {h.id.substring(0,8)}</span>
              </div>
              <div className="flex justify-between items-end">
                  <span className="text-sm text-slate-300 font-medium truncate max-w-[150px]">{h.filename}</span>
                  <div className="text-right">
                      <div className="text-xs font-bold text-purple-400">{h.best_model}</div>
                      <div className="text-[10px] text-slate-500">R²: <span className="font-mono text-white">{h.score.toFixed(4)}</span></div>
                  </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
