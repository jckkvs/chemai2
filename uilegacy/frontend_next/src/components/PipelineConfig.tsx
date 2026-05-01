// src/components/PipelineConfig.tsx
'use client'

import { useChemAIStore } from '@/lib/store'
import { runPipeline } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Settings2, Sliders } from 'lucide-react'

export default function PipelineConfig() {
  const { pipelineConfig, updatePipelineConfig, setLoading, isLoading, setAnalysisResult, setError } = useChemAIStore()

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await runPipeline(pipelineConfig)
      setAnalysisResult(result)
    } catch (err: any) {
      setError(err.response?.data?.detail || '解析に失敗しました')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">⚙️ パイプライン設定</h2>
        <button
          onClick={handleRun}
          disabled={isLoading}
          className="flex items-center gap-2 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors font-semibold shadow-md"
        >
          {isLoading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
          解析開始
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Basic Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="w-5 h-5" />
              基本設定
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">交差検証 (CV Folds)</label>
              <input 
                type="number" 
                value={pipelineConfig.cv_folds}
                onChange={(e) => updatePipelineConfig({ cv_folds: parseInt(e.target.value) })}
                className="w-full p-2 border rounded-md"
                min="2" max="10"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">数値スケーラー</label>
              <select 
                value={pipelineConfig.num_scaler}
                onChange={(e) => updatePipelineConfig({ num_scaler: e.target.value as any })}
                className="w-full p-2 border rounded-md"
              >
                <option value="standard">StandardScaler</option>
                <option value="robust">RobustScaler</option>
                <option value="minmax">MinMaxScaler</option>
                <option value="none">なし</option>
              </select>
            </div>
          </CardContent>
        </Card>

        {/* Feature Engineering */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sliders className="w-5 h-5" />
              特徴量生成・選択
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <input 
                type="checkbox" 
                checked={pipelineConfig.do_polynomial}
                onChange={(e) => updatePipelineConfig({ do_polynomial: e.target.checked })}
                className="w-4 h-4 text-blue-600"
              />
              <label className="text-sm font-medium text-slate-700">多項式特徴量 (Polynomial)</label>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">特徴量選択手法</label>
              <select 
                value={pipelineConfig.feature_selector}
                onChange={(e) => updatePipelineConfig({ feature_selector: e.target.value as any })}
                className="w-full p-2 border rounded-md"
              >
                <option value="none">すべて使用</option>
                <option value="variance">分散閾値</option>
                <option value="select_from_model_rf">RandomForest 重要度</option>
              </select>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Simple fallback for icons
function RefreshCw({ className }: { className?: string }) {
  return <div className={className}>↻</div>
}

// Simple fallback for missing UI components
function Card({ children }: { children: React.ReactNode }) {
  return <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">{children}</div>
}
function CardHeader({ children }: { children: React.ReactNode }) {
  return <div className="p-4 border-b border-slate-100 font-semibold">{children}</div>
}
function CardTitle({ children }: { children: React.ReactNode }) {
  return <div className="text-lg">{children}</div>
}
function CardContent({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`p-6 ${className}`}>{children}</div>
}
