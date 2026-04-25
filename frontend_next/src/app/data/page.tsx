// src/app/data/page.tsx
'use client'

import { useState, useRef } from 'react'
import { useChemAIStore } from '@/lib/store'
import { uploadData, getDataInfo } from '@/lib/api'
import { useQuery } from '@tanstack/react-query'
import { Upload, FileText, AlertCircle } from 'lucide-react'

export default function DataPage() {
  const { setLoadedData, error, setError, setLoading } = useChemAIStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  // Fetch data info on mount
  const { data: dataInfo, refetch } = useQuery({
    queryKey: ['dataInfo'],
    queryFn: getDataInfo,
    enabled: false,
    retry: false,
  })

  const handleFileSelect = async (file: File) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await uploadData(file)
      
      if (response.success) {
        setLoadedData({
          filename: response.filename,
          df: response.preview,
          columns: response.columns,
          targetCol: response.target_col,
          taskType: response.task_type,
          metrics: response.metrics,
        })
        refetch()
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'アップロードに失敗しました')
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">📂 データ読込</h1>

      {/* Upload Zone */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 mb-8 overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center gap-2 font-semibold">
          <Upload className="w-5 h-5 text-blue-500" />
          CSV / Excel アップロード
        </div>
        <div className="p-6">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              dragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-slate-400'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            />
            <FileText className="w-12 h-12 mx-auto mb-4 text-slate-400" />
            <p className="text-slate-600 mb-2">
              ファイルをドラッグ＆ドロップ、またはクリックして選択
            </p>
            <p className="text-sm text-slate-400">
              対応形式: CSV, Excel (.xlsx, .xls)
            </p>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Data Preview */}
      {dataInfo && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="p-4 border-b border-slate-100 font-semibold">
            読み込んだデータ: {dataInfo.filename}
          </div>
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <Metric label="行数" value={dataInfo.metrics.rows.toLocaleString()} />
              <Metric label="列数" value={dataInfo.metrics.cols} />
              <Metric label="欠損率" value={`${(dataInfo.metrics.missing_rate * 100).toFixed(1)}%`} />
              <Metric label="数値列" value={dataInfo.metrics.numeric_cols} />
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    {dataInfo.columns.map((col) => (
                      <th key={col} className="text-left py-2 px-3 font-medium">
                        {col}
                        {col === dataInfo.target_col && (
                          <span className="ml-1 px-1.5 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                            目的変数
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataInfo.preview.map((row, idx) => (
                    <tr key={idx} className="border-b hover:bg-slate-50">
                      {dataInfo.columns.map((col) => (
                        <td key={col} className="py-2 px-3">
                          {row[col] === null ? '—' : typeof row[col] === 'number' ? row[col].toFixed(4) : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="p-4 bg-slate-50 rounded-lg text-center">
      <div className="text-2xl font-bold text-slate-800">{value}</div>
      <div className="text-sm text-slate-500">{label}</div>
    </div>
  )
}
