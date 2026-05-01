// frontend_next/src/app/data/page.tsx
'use client';

import { useState, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useChemAIStore } from '@/lib/store';
import { uploadData, getDataInfo, getBenchmarks, loadBenchmark } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Upload, FileText, Database, AlertCircle, CheckCircle, Loader2, Download } from 'lucide-react';

export default function DataPage() {
  const { setLoadedData, error, setError, setLoading, setActiveTab } = useChemAIStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);

  const { data: dataInfo, refetch: refetchDataInfo } = useQuery({
    queryKey: ['dataInfo'],
    queryFn: getDataInfo,
    enabled: false,
    retry: false,
  });

  const { data: benchmarks } = useQuery({
    queryKey: ['benchmarks'],
    queryFn: getBenchmarks,
    staleTime: 10 * 60 * 1000,
  });

  const handleFileSelect = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setUploadProgress(0);
    try {
      const response = await uploadData(file, (percent) => setUploadProgress(percent));
      if (response.success) {
        setLoadedData({
          filename: response.filename,
          columns: response.column_details || response.columns.map((name: string) => ({ name, type: 'numeric' })),
          targetCol: response.target_col,
          taskType: response.task_type,
          preview: response.preview,
          metrics: response.metrics,
        });
        await refetchDataInfo();
        setActiveTab('eda');
      }
    } catch (err: any) {
      setError(err.message || 'アップロードに失敗しました');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  }, [setLoadedData, setError, setLoading, refetchDataInfo, setActiveTab]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleBenchmarkLoad = async (datasetId: string) => {
    setLoading(true);
    setError(null);
    setSelectedBenchmark(datasetId);
    try {
      const response = await loadBenchmark(datasetId);
      if (response.success) {
        setLoadedData({
          filename: response.filename,
          columns: response.column_details || response.columns.map((name: string) => ({ name, type: 'numeric' })),
          targetCol: response.target_col,
          taskType: response.task_type,
          preview: response.preview,
          metrics: response.metrics,
        });
        await refetchDataInfo();
        setActiveTab('eda');
      }
    } catch (err: any) {
      setError(err.message || 'ベンチマークの読み込みに失敗しました');
    } finally {
      setLoading(false);
      setSelectedBenchmark(null);
    }
  };

  const handleDownloadSample = () => {
    const headers = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Target'];
    const rows = Array.from({ length: 10 }, (_, i) => 
      headers.map((_, j) => (Math.random() * 100).toFixed(2)).join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Data Management</h1>
          <p className="text-slate-500 mt-1">Upload your datasets or select from benchmarks.</p>
        </div>
        <Button variant="outline" size="sm" onClick={handleDownloadSample}>
          <Download className="w-4 h-4 mr-2" />
          Sample CSV
        </Button>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          <Card className="border-2 border-dashed border-slate-200 shadow-none hover:border-indigo-300 transition-colors">
            <CardContent className="pt-12 pb-12 text-center">
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                />
                <div className={`w-20 h-20 rounded-3xl mx-auto mb-6 flex items-center justify-center transition-colors ${dragOver ? 'bg-indigo-600 text-white' : 'bg-indigo-50 text-indigo-600'}`}>
                  <Upload size={32} />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">Drag & Drop Files</h3>
                <p className="text-slate-400 text-sm max-w-xs mx-auto mb-6">
                  CSV, Excel (.xlsx, .xls) files are supported. Max size 50MB.
                </p>
                <Button variant="secondary" className="font-bold">
                  Browse Files
                </Button>
              </div>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <div className="mt-8 max-w-xs mx-auto">
                  <div className="flex justify-between text-xs font-bold text-slate-400 mb-2 uppercase">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-1.5">
                    <div 
                      className="bg-indigo-600 h-1.5 rounded-full transition-all duration-300 shadow-sm"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {dataInfo && (
            <Card className="overflow-hidden border-slate-200">
              <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                <CardTitle className="text-lg">Data Preview: {dataInfo.filename}</CardTitle>
                <CardDescription>
                  Showing first 8 rows of {dataInfo.metrics.rows.toLocaleString()} entries.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow className="hover:bg-transparent">
                        {dataInfo.columns.map((col) => (
                          <TableHead key={col} className="bg-slate-50/30 whitespace-nowrap font-bold text-slate-700">
                            {col}
                            {col === dataInfo.target_col && (
                              <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded-full bg-indigo-100 text-indigo-700 text-[10px] uppercase tracking-tighter">
                                Target
                              </span>
                            )}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataInfo.preview.map((row, idx) => (
                        <TableRow key={idx} className="hover:bg-indigo-50/30 transition-colors">
                          {dataInfo.columns.map((col) => {
                            const value = row[col];
                            return (
                              <TableCell key={col} className="text-xs font-medium text-slate-600 whitespace-nowrap">
                                {value === null || value === undefined ? (
                                  <span className="text-slate-300">null</span>
                                ) : typeof value === 'number' ? (
                                  value.toLocaleString(undefined, { maximumFractionDigits: 4 })
                                ) : (
                                  String(value)
                                )}
                              </TableCell>
                            );
                          })}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-8">
          <Card className="border-slate-200">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Database className="text-indigo-600 w-5 h-5" />
                Benchmarks
              </CardTitle>
              <CardDescription>Pre-loaded chemical datasets.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              {benchmarks?.map((benchmark) => (
                <div 
                  key={benchmark.id}
                  onClick={() => handleBenchmarkLoad(benchmark.id)}
                  className={`group p-4 rounded-xl border transition-all cursor-pointer ${
                    selectedBenchmark === benchmark.id 
                      ? 'border-indigo-600 bg-indigo-50 shadow-md shadow-indigo-100' 
                      : 'border-slate-100 hover:border-indigo-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-bold text-slate-800">{benchmark.name}</h4>
                    {selectedBenchmark === benchmark.id ? (
                      <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                    ) : (
                      <ArrowRight className="w-4 h-4 text-slate-300 group-hover:translate-x-1 transition-transform" />
                    )}
                  </div>
                  <p className="text-xs text-slate-500 line-clamp-2">{benchmark.description}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          {dataInfo && (
            <Card className="border-slate-200 bg-indigo-600 text-white shadow-xl shadow-indigo-100">
              <CardContent className="pt-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-3 bg-white/20 rounded-2xl backdrop-blur-sm">
                    <CheckCircle className="w-8 h-8" />
                  </div>
                  <div>
                    <p className="text-white/60 text-xs font-bold uppercase tracking-widest">Ready for analysis</p>
                    <h3 className="text-xl font-bold">Data Loaded</h3>
                  </div>
                </div>
                <div className="space-y-4 mb-8">
                  <div className="flex justify-between text-sm border-b border-white/10 pb-2">
                    <span className="text-white/60">Entries</span>
                    <span className="font-bold">{dataInfo.metrics.rows.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm border-b border-white/10 pb-2">
                    <span className="text-white/60">Features</span>
                    <span className="font-bold">{dataInfo.metrics.cols}</span>
                  </div>
                  <div className="flex justify-between text-sm border-b border-white/10 pb-2">
                    <span className="text-white/60">Missing Values</span>
                    <span className="font-bold">{(dataInfo.metrics.missing_rate * 100).toFixed(1)}%</span>
                  </div>
                </div>
                <Button 
                  className="w-full bg-white text-indigo-600 hover:bg-slate-100 h-12 font-bold"
                  onClick={() => setActiveTab('eda')}
                >
                  Go to EDA
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-8 p-4 bg-red-50 border border-red-100 rounded-2xl flex items-center gap-3 text-red-700 animate-in fade-in zoom-in duration-300">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}
    </div>
  );
}
