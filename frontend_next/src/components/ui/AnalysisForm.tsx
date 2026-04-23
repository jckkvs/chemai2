import React from 'react';
import { Play, Settings2, Loader2 } from 'lucide-react';

interface AnalysisFormProps {
  config: any;
  isRunning: boolean;
  onConfigChange: (key: string, value: any) => void;
  onStart: () => void;
}

export const AnalysisForm: React.FC<AnalysisFormProps> = ({ 
  config, 
  isRunning, 
  onConfigChange, 
  onStart 
}) => {
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6 space-y-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Settings2 className="w-5 h-5 text-cyan-400" />
          解析設定
        </h2>
      </div>

      {/* CV 設定 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">CV分割数</label>
          <input
            type="number"
            value={config.cv_folds || 5}
            onChange={(e) => onConfigChange('cv_folds', parseInt(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            min={2}
            max={10}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">スケーラー</label>
          <select
            value={config.num_scaler || 'standard'}
            onChange={(e) => onConfigChange('num_scaler', e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
          >
            <option value="standard">StandardScaler</option>
            <option value="robust">RobustScaler</option>
            <option value="minmax">MinMaxScaler</option>
            <option value="none">なし</option>
          </select>
        </div>
      </div>

      {/* モデル選択 (簡易チェックボックス群) */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">使用するモデル</label>
        <div className="flex flex-wrap gap-2">
          {['RandomForest', 'XGBoost', 'LightGBM', 'SVR', 'Ridge', 'Lasso'].map((model) => (
            <label key={model} className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.selected_models?.includes(model)}
                onChange={(e) => {
                  const current = config.selected_models || [];
                  const updated = e.target.checked 
                    ? [...current, model] 
                    : current.filter((m: string) => m !== model);
                  onConfigChange('selected_models', updated);
                }}
                className="form-checkbox h-4 w-4 text-cyan-500 rounded border-slate-700 bg-slate-900 focus:ring-cyan-500"
              />
              <span className="text-sm text-slate-300">{model}</span>
            </label>
          ))}
        </div>
      </div>

      {/* 実行ボタン */}
      <button
        onClick={onStart}
        disabled={isRunning}
        className={`
          w-full flex items-center justify-center gap-2 py-3 rounded-lg font-bold text-white transition-all
          ${isRunning 
            ? 'bg-slate-700 cursor-not-allowed' 
            : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 shadow-lg hover:shadow-cyan-500/20'}
        `}
      >
        {isRunning ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            解析実行中...
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            解析開始
          </>
        )}
      </button>
    </div>
  );
};
