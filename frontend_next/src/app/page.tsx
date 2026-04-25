// src/app/page.tsx
'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useChemAIStore } from '@/lib/store'
import { initSession } from '@/lib/api'

export default function Home() {
  const router = useRouter()
  const { sessionId, setSessionId } = useChemAIStore()

  useEffect(() => {
    const setupSession = async () => {
      if (!sessionId) {
        try {
          const id = await initSession()
          setSessionId(id)
        } catch (error) {
          console.error('Failed to initialize session:', error)
        }
      }
    }
    setupSession()
  }, [sessionId, setSessionId])

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-12">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-800 mb-4">
            🧪 ChemAI Nexus
          </h1>
          <p className="text-lg text-slate-600">
            化学構造データと機械学習を統合した解析プラットフォーム
          </p>
        </header>

        <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          <Card
            title="📂 データ読込"
            description="CSV/Excel ファイルをアップロードして解析を開始"
            onClick={() => router.push('/data')}
          />
          <Card
            title="⚙️ 機械学習"
            description="AutoML・前処理・特徴量選択をワンクリックで"
            onClick={() => router.push('/pipeline')}
          />
          <Card
            title="📊 結果・レポート"
            description="モデル評価・可視化・レポート出力"
            onClick={() => router.push('/results')}
          />
        </div>
      </div>
    </main>
  )
}

function Card({ title, description, onClick }: {
  title: string
  description: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="p-6 bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow text-left border border-slate-200"
    >
      <h3 className="text-xl font-semibold text-slate-800 mb-2">{title}</h3>
      <p className="text-slate-600">{description}</p>
    </button>
  )
}
