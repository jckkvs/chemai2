// src/components/ResultsView.tsx
'use client'

import { useQuery } from '@tanstack/react-query'
import { getResults } from '@/lib/api'
import { useChemAIStore } from '@/lib/store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RefreshCw, TrendingUp, Award, Clock } from 'lucide-react'

export default function ResultsView() {
  const { analysisResult, setAnalysisResult } = useChemAIStore()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['results'],
    queryFn: getResults,
    onSuccess: (data) => setAnalysisResult(data),
  })

  const result = analysisResult || data

  if (!result || result.status === 'pending') {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-slate-400">
        <Clock className="w-12 h-12 mb-4 opacity-20" />
        <p>解析結果はまだありません。パイプラインを実行してください。</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">📈 解析レポート</h2>
        <button 
          onClick={() => refetch()} 
          className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          結果を更新
        </button>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <MetricCard 
          icon={<Award className="w-5 h-5 text-yellow-500" />}
          label="最良モデル"
          value={result.best_model || '—'}
        />
        <MetricCard 
          icon={<TrendingUp className="w-5 h-5 text-green-500" />}
          label="評価スコア (R²/Acc)"
          value={result.score?.toFixed(4) || '—'}
        />
        <MetricCard 
          icon={<Clock className="w-5 h-5 text-blue-500" />}
          label="ステータス"
          value={result.status.toUpperCase()}
          valueClassName="text-green-600"
        />
      </div>

      {/* Feature Importance */}
      {result.feature_importances && (
        <Card>
          <CardHeader>
            <CardTitle>🔥 重要特徴量 (Top 10)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {result.feature_importances.slice(0, 10).map((feat, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-slate-700">{feat.name}</span>
                    <span className="text-slate-500">{feat.value.toFixed(4)}</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div 
                      className="bg-blue-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(100, (feat.value / result.feature_importances![0].value) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function MetricCard({ icon, label, value, valueClassName = '' }: {
  icon: React.ReactNode
  label: string
  value: string | number
  valueClassName?: string
}) {
  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <span className="text-sm font-medium text-slate-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-2xl font-bold text-slate-800 ${valueClassName}`}>{value}</div>
    </div>
  )
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
function CardContent({ children }: { children: React.ReactNode }) {
  return <div className="p-6">{children}</div>
}
