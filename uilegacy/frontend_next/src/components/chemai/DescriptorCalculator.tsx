'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Play, Settings, CheckCircle, AlertCircle } from 'lucide-react';
import { chemaiClient, TaskProgress } from '@/api/chemai-client';
// Note: We'll use a generic toast if use-toast is not available, or just alert for now.
// For the final implementation, let's assume a basic toast hook exists or just use a fallback.

interface DescriptorCalculatorProps {
  dataId: string;
  smilesColumns: string[];
  onCalculationComplete?: (resultId: string) => void;
}

interface EngineInfo {
  name: string;
  available: boolean;
  description: string;
}

export function DescriptorCalculator({ dataId, smilesColumns, onCalculationComplete }: DescriptorCalculatorProps) {
  // Form state
  const [selectedSmilesColumn, setSelectedSmilesColumn] = useState<string>(smilesColumns[0] || '');
  const [selectedEngines, setSelectedEngines] = useState<string[]>([]);
  const [charge, setCharge] = useState<number>(0);
  const [multiplicity, setMultiplicity] = useState<number>(1);
  const [pH, setPH] = useState<string>('');
  
  // Engine list state
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [loadingEngines, setLoadingEngines] = useState(true);
  
  // Calculation state
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Load available engines on mount
  useEffect(() => {
    const loadEngines = async () => {
      try {
        const response = await chemaiClient.getAvailableEngines();
        setEngines(response.engines);
        // Pre-select available engines
        const available = response.engines.filter(e => e.available).map(e => e.name);
        setSelectedEngines(available);
      } catch (err: any) {
        console.error("Failed to load engines", err);
      } finally {
        setLoadingEngines(false);
      }
    };
    loadEngines();
  }, []);
  
  // Subscribe to WebSocket progress updates
  useEffect(() => {
    if (!taskId) return;
    
    const unsubscribe = chemaiClient.subscribeToProgress(taskId, (progress: TaskProgress) => {
      setProgress(progress);
      
      if (progress.type === 'complete') {
        setIsCalculating(false);
        if (progress.result?.result_id && onCalculationComplete) {
          onCalculationComplete(progress.result.result_id);
        }
      } else if (progress.type === 'error') {
        setIsCalculating(false);
        setError(progress.result?.error || '計算中にエラーが発生しました');
      }
    });
    
    return () => unsubscribe();
  }, [taskId, onCalculationComplete]);
  
  // Handle engine selection toggle
  const toggleEngine = useCallback((engineName: string) => {
    setSelectedEngines(prev => 
      prev.includes(engineName) 
        ? prev.filter(e => e !== engineName)
        : [...prev, engineName]
    );
  }, []);
  
  // Start descriptor calculation
  const handleCalculate = async () => {
    if (!selectedSmilesColumn) return;
    if (selectedEngines.length === 0) return;
    
    setIsCalculating(true);
    setError(null);
    setProgress(null);
    
    try {
      const chargeConfig = {
        charge,
        multiplicity,
        pH: pH ? parseFloat(pH) : undefined,
      };
      
      const response = await chemaiClient.calculateDescriptors(
        dataId,
        selectedSmilesColumn,
        selectedEngines,
        chargeConfig
      );
      
      setTaskId(response.task_id);
    } catch (err: any) {
      setIsCalculating(false);
      setError(err.detail || err.error || '計算の開始に失敗しました');
    }
  };
  
  // Cancel ongoing calculation
  const handleCancel = async () => {
    if (!taskId) return;
    try {
      await chemaiClient.cancelTask(taskId);
      setIsCalculating(false);
      setTaskId(null);
    } catch (err: any) {
      console.error("Cancel failed", err);
    }
  };
  
  return (
    <Card className="w-full shadow-lg border-none">
      <CardHeader className="bg-slate-50 border-b">
        <CardTitle className="flex items-center gap-2 text-slate-800">
          <Settings className="w-5 h-5 text-indigo-500" />
          化学記述子計算 (Advanced Plugins)
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6 space-y-6">
        
        {/* SMILES Column Selection */}
        <div className="space-y-2">
          <Label htmlFor="smiles-column" className="text-xs font-bold uppercase text-slate-400">SMILES列</Label>
          <Select value={selectedSmilesColumn} onValueChange={setSelectedSmilesColumn}>
            <SelectTrigger id="smiles-column" className="rounded-xl">
              <SelectValue placeholder="SMILESを含む列を選択" />
            </SelectTrigger>
            <SelectContent>
              {smilesColumns.map(col => (
                <SelectItem key={col} value={col}>{col}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        {/* Engine Selection */}
        <div className="space-y-3">
          <Label className="text-xs font-bold uppercase text-slate-400">記述子エンジン</Label>
          {loadingEngines ? (
            <div className="flex items-center gap-2 text-sm text-slate-400 py-4">
              <Loader2 className="w-4 h-4 animate-spin" />
              エンジン一覧を読み込み中...
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {engines.map(engine => (
                <div 
                  key={engine.name}
                  className={`flex items-start space-x-3 p-4 rounded-2xl border transition-all ${
                    !engine.available ? 'opacity-40 bg-slate-50 border-transparent' : 'hover:border-indigo-200 hover:bg-slate-50 border-slate-100'
                  }`}
                >
                  <Checkbox
                    id={`engine-${engine.name}`}
                    checked={selectedEngines.includes(engine.name)}
                    onCheckedChange={() => toggleEngine(engine.name)}
                    disabled={!engine.available || isCalculating}
                    className="mt-1"
                  />
                  <div className="space-y-1">
                    <Label 
                      htmlFor={`engine-${engine.name}`}
                      className="font-bold text-slate-700 cursor-pointer flex items-center"
                    >
                      {engine.name}
                      {!engine.available && (
                        <Badge variant="secondary" className="ml-2 text-[10px]">UNAVAILABLE</Badge>
                      )}
                    </Label>
                    <p className="text-[11px] text-slate-500 leading-tight">{engine.description}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Charge & Multiplicity Settings */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 bg-slate-50 p-5 rounded-2xl border border-slate-100">
          <div className="space-y-1.5">
            <Label htmlFor="charge" className="text-[10px] font-bold text-slate-400 uppercase">形式電荷</Label>
            <Input
              id="charge"
              type="number"
              value={charge}
              onChange={(e) => setCharge(parseInt(e.target.value) || 0)}
              disabled={isCalculating}
              className="h-9 rounded-lg"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="multiplicity" className="text-[10px] font-bold text-slate-400 uppercase">多重度</Label>
            <Input
              id="multiplicity"
              type="number"
              value={multiplicity}
              onChange={(e) => setMultiplicity(parseInt(e.target.value) || 1)}
              disabled={isCalculating}
              className="h-9 rounded-lg"
            />
          </div>
          <div className="space-y-1.5 md:col-span-2">
            <Label htmlFor="ph" className="text-[10px] font-bold text-slate-400 uppercase">pH (オプション)</Label>
            <Input
              id="ph"
              type="number"
              step="0.1"
              value={pH}
              onChange={(e) => setPH(e.target.value)}
              placeholder="例: 7.4"
              disabled={isCalculating}
              className="h-9 rounded-lg"
            />
          </div>
        </div>
        
        {/* Error Display */}
        {error && (
          <Alert variant="destructive" className="rounded-xl border-none bg-red-50 text-red-600">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs font-medium">{error}</AlertDescription>
          </Alert>
        )}
        
        {/* Progress Display */}
        {isCalculating && progress && (
          <div className="space-y-3 p-4 bg-indigo-50 rounded-2xl border border-indigo-100">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-indigo-700">{progress.message}</span>
              <span className="font-black text-indigo-800">{Math.round(progress.progress)}%</span>
            </div>
            <Progress value={progress.progress} className="h-1.5 bg-indigo-200" />
            {progress.data?.current_engine && (
              <p className="text-[10px] text-indigo-400">
                Processing: <span className="font-bold">{progress.data.current_engine}</span>
              </p>
            )}
          </div>
        )}
        
        {/* Action Buttons */}
        <div className="flex gap-3 pt-2">
          <Button 
            onClick={handleCalculate}
            disabled={isCalculating || loadingEngines || selectedEngines.length === 0 || !selectedSmilesColumn}
            className="flex-1 rounded-xl h-12 bg-slate-900 hover:bg-slate-800 shadow-lg shadow-slate-200"
          >
            {isCalculating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                計算中...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4 fill-current" />
                記述子計算を実行
              </>
            )}
          </Button>
          
          {isCalculating && (
            <Button variant="outline" onClick={handleCancel} className="rounded-xl h-12 border-slate-200">
              キャンセル
            </Button>
          )}
          
          {!isCalculating && progress?.type === 'complete' && (
            <Badge variant="success" className="flex items-center gap-1 px-4 py-2 rounded-xl border-none">
              <CheckCircle className="w-3 h-3" />
              完了
            </Badge>
          )}
        </div>
        
      </CardContent>
    </Card>
  );
}
