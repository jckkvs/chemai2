import React, { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { api } from '../../lib/api';

interface DataUploaderProps {
  onDataLoaded: (data: any) => void;
}

export const DataUploader: React.FC<DataUploaderProps> = ({ onDataLoaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // FastAPI のアップロードエンドポイントを叩く
      const response = await api.post('/api/data/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // 成功時に親コンポーネントにデータとファイルオブジェクトを渡す
      onDataLoaded({
        file, // 元の File オブジェクトを保持
        metadata: {
          filename: file.name,
          columns: response.data.column_names,
          preview: response.data.preview,
          rows: response.data.rows,
          // ... その他のメタデータ
        }
      });
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ファイルのアップロードに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300
          ${isDragging ? 'border-cyan-400 bg-cyan-400/10' : 'border-slate-700 hover:border-slate-500 bg-slate-800/50'}
        `}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={loading}
        />
        
        {loading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-lg font-medium text-white">処理中...</p>
          </div>
        ) : (
          <>
            <Upload className="w-16 h-16 text-slate-400 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">ファイルをドロップまたはクリック</h3>
            <p className="text-slate-400 mb-4">CSV または Excel ファイル (.csv, .xlsx, .xls)</p>
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3 text-red-400">
          <AlertCircle className="w-5 h-5" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
