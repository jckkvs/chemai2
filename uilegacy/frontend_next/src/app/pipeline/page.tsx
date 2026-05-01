// frontend_next/src/app/pipeline/page.tsx
'use client';

import { useState, useEffect, useMemo } from 'react';
import { useForm, Controller, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useChemAIStore } from '@/lib/store';
import { 
  getPipelineConfig, updatePipelineConfig, runPipeline, 
  getAvailableModels
} from '@/lib/api';
import { getEstimatorSchema, featureEngineSchemas } from '@/config/estimatorSchemas';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { AlertCircle, Play, Save, RefreshCw, Plus, Trash2, Loader2, Settings2 } from 'lucide-react';
import { useRouter } from 'next/navigation';

// Zod Schema for Pipeline Configuration
const pipelineSchema = z.object({
  preprocessing: z.object({
    num_scaler: z.enum(['standard', 'robust', 'minmax', 'maxabs', 'none']),
    num_imputer: z.enum(['median', 'mean', 'knn', 'iterative', 'drop']),
    num_transform: z.enum(['none', 'boxcox', 'yeojohnson', 'quantile_uniform', 'quantile_normal', 'log1p']),
    cat_encoder: z.enum(['onehot', 'ordinal', 'target', 'binary', 'leave_one_out']),
    cat_imputer: z.enum(['most_frequent', 'constant', 'drop']),
    exclude_cols: z.array(z.string()).default([])
  }),
  feature_generation: z.object({
    do_polynomial: z.boolean(),
    poly_degree: z.coerce.number().min(2).max(4),
    poly_interaction_only: z.boolean(),
    engines: z.array(z.object({
      key: z.string(),
      enabled: z.boolean(),
      params: z.record(z.any()).optional()
    })).default([])
  }),
  feature_selection: z.object({
    feature_selector: z.enum(['none', 'variance', 'selectkbest_f', 'selectkbest_mi', 'select_from_model_lasso', 'select_from_model_rf', 'rfe', 'boruta']),
    n_features_to_select: z.coerce.number().min(1).max(500)
  }),
  model: z.object({
    estimator: z.string().min(1),
    params: z.record(z.any()).default({})
  }),
  constraints: z.array(z.object({
    feature: z.string(),
    direction: z.coerce.number().pipe(z.literal(-1).or(z.literal(0)).or(z.literal(1))),
    strength: z.enum(['hard', 'soft']),
    sigma_range: z.coerce.number().default(3.0),
    linear: z.boolean()
  })).default([]),
  execution: z.object({
    cv_strategy: z.enum(['kfold', 'stratified', 'group', 'time_series', 'loo', 'shuffle']),
    cv_folds: z.coerce.number().min(2).max(20),
    do_eda: z.boolean(),
    do_shap: z.boolean()
  })
});

type PipelineFormValues = z.infer<typeof pipelineSchema>;

export default function PipelinePage() {
  const router = useRouter();
  const { sessionId, targetCol, taskType, columns, error: storeError, setError } = useChemAIStore();
  const [activeTab, setActiveTabState] = useState('preprocessing');
  const [selectedEngine, setSelectedEngine] = useState<string>('');

  const { data: availableEstimators } = useQuery({
    queryKey: ['estimators', taskType],
    queryFn: () => getAvailableModels(taskType || 'regression'),
    enabled: !!taskType && !!sessionId
  });

  const { data: pipelineConfig, isLoading: configLoading } = useQuery({
    queryKey: ['pipelineConfig', sessionId],
    queryFn: () => getPipelineConfig(),
    enabled: !!sessionId
  });

  const form = useForm<PipelineFormValues>({
    resolver: zodResolver(pipelineSchema),
    defaultValues: {
      preprocessing: { num_scaler: 'standard', num_imputer: 'median', num_transform: 'none', cat_encoder: 'onehot', cat_imputer: 'most_frequent', exclude_cols: [] },
      feature_generation: { do_polynomial: false, poly_degree: 2, poly_interaction_only: true, engines: [] },
      feature_selection: { feature_selector: 'none', n_features_to_select: 20 },
      model: { estimator: 'RandomForestRegressor', params: {} },
      constraints: [],
      execution: { cv_strategy: 'kfold', cv_folds: 5, do_eda: true, do_shap: true }
    }
  });

  const { fields: engineFields, append: appendEngine, remove: removeEngine } = useFieldArray({
    control: form.control,
    name: "feature_generation.engines"
  });

  useEffect(() => {
    if (pipelineConfig) {
      // Mapping API config to Form values if they differ slightly
      form.reset({
        preprocessing: pipelineConfig.preprocessing || form.getValues('preprocessing'),
        feature_generation: pipelineConfig.feature_generation || form.getValues('feature_generation'),
        feature_selection: pipelineConfig.feature_selection || form.getValues('feature_selection'),
        model: {
          estimator: pipelineConfig.estimator || 'RandomForestRegressor',
          params: pipelineConfig.estimator_params || {}
        },
        constraints: (pipelineConfig.monotonic_constraints || []).map((c: any) => ({
          feature: c.feature,
          direction: c.direction,
          strength: c.strength,
          sigma_range: c.sigma_range,
          linear: c.linear || false
        })),
        execution: {
            cv_strategy: (pipelineConfig.cv_strategy as any) || 'kfold',
            cv_folds: pipelineConfig.cv_folds || 5,
            do_eda: pipelineConfig.do_eda ?? true,
            do_shap: pipelineConfig.do_shap ?? true
        }
      });
    }
  }, [pipelineConfig, form]);

  const saveMutation = useMutation({
    mutationFn: (data: PipelineFormValues) => updatePipelineConfig({
        ...data,
        estimator: data.model.estimator,
        estimator_params: data.model.params,
        monotonic_constraints: data.constraints
    }),
    onSuccess: () => alert('設定を保存しました'),
    onError: (e: any) => setError(e.message)
  });

  const runMutation = useMutation({
    mutationFn: (data: PipelineFormValues) => runPipeline({
        ...data,
        estimator: data.model.estimator,
        estimator_params: data.model.params,
        monotonic_constraints: data.constraints
    } as any),
    onSuccess: () => {
      router.push('/results');
    },
    onError: (e: any) => setError(e.message)
  });

  const estimatorSchema = useMemo(() => {
    const est = form.watch('model.estimator');
    return est ? getEstimatorSchema(est, taskType || 'regression') : null;
  }, [form.watch('model.estimator'), taskType]);

  const addEngine = () => {
    if (!selectedEngine) return;
    const schema = featureEngineSchemas[selectedEngine];
    appendEngine({ 
        key: selectedEngine, 
        enabled: true, 
        params: schema?.params ? Object.fromEntries(
            Object.entries(schema.params).map(([k, v]: [string, any]) => [k, v.default])
        ) : {} 
    });
    setSelectedEngine('');
  };

  const numericCols = columns.filter(c => c.type === 'numeric' || c.type === 'binary').map(c => c.name);

  if (!sessionId) return <div className="p-12 text-center text-slate-500">セッションが見つかりません。データをロードしてください。</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">⚙️ パイプライン・ビルダー</h1>
          <p className="text-slate-500 mt-1">
            {targetCol ? `Target: ${targetCol} (${taskType})` : 'ターゲット変数を設定してください'}
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" onClick={() => form.reset()} disabled={saveMutation.isPending || runMutation.isPending}>
            <RefreshCw className="w-4 h-4 mr-2" />リセット
          </Button>
          <Button variant="outline" onClick={() => saveMutation.mutate(form.getValues())} disabled={saveMutation.isPending || runMutation.isPending}>
            <Save className="w-4 h-4 mr-2" />構成を保存
          </Button>
          <Button 
            onClick={() => runMutation.mutate(form.getValues())} 
            disabled={runMutation.isPending || !targetCol} 
            className="bg-indigo-600 hover:bg-indigo-700 shadow-md transition-all active:scale-95"
          >
            {runMutation.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
            解析実行
          </Button>
        </div>
      </div>

      {(storeError || runMutation.error || saveMutation.error) && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm">{String(storeError || runMutation.error || saveMutation.error)}</p>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTabState} className="space-y-6">
        <TabsList className="bg-slate-100 p-1 rounded-xl w-full md:w-auto overflow-x-auto flex whitespace-nowrap">
          <TabsTrigger value="preprocessing" className="rounded-lg px-6">前処理</TabsTrigger>
          <TabsTrigger value="features" className="rounded-lg px-6">特徴量生成</TabsTrigger>
          <TabsTrigger value="model" className="rounded-lg px-6">モデル設定</TabsTrigger>
          <TabsTrigger value="constraints" className="rounded-lg px-6">物理制約</TabsTrigger>
          <TabsTrigger value="execution" className="rounded-lg px-6">実行・バリデーション</TabsTrigger>
        </TabsList>

        {/* --- Preprocessing Tab --- */}
        <TabsContent value="preprocessing">
          <Card className="border-none shadow-lg">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg">データクリーニング & 正規化</CardTitle>
              <CardDescription>欠損値の補完、スケーリング、非線形変換を設定します。</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-8">
              <div className="grid md:grid-cols-2 gap-10">
                <div className="space-y-6">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-6 bg-blue-500 rounded-full" />
                    <h3 className="font-bold text-slate-800">数値列 (Numeric)</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-1.5">
                      <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">スケーラー</Label>
                      <Controller name="preprocessing.num_scaler" control={form.control} render={({ field }) => (
                        <Select onValueChange={field.onChange} value={field.value}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="standard">StandardScaler (Z-Score)</SelectItem>
                            <SelectItem value="robust">RobustScaler (IQR based)</SelectItem>
                            <SelectItem value="minmax">MinMaxScaler [0, 1]</SelectItem>
                            <SelectItem value="maxabs">MaxAbsScaler [-1, 1]</SelectItem>
                            <SelectItem value="none">なし</SelectItem>
                          </SelectContent>
                        </Select>
                      )} />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">欠損値補完</Label>
                      <Controller name="preprocessing.num_imputer" control={form.control} render={({ field }) => (
                        <Select onValueChange={field.onChange} value={field.value}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="median">中央値 (Median)</SelectItem>
                            <SelectItem value="mean">平均値 (Mean)</SelectItem>
                            <SelectItem value="knn">KNN Imputer</SelectItem>
                            <SelectItem value="iterative">Iterative Imputer (MICE)</SelectItem>
                            <SelectItem value="drop">行削除</SelectItem>
                          </SelectContent>
                        </Select>
                      )} />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">非線形変換</Label>
                      <Controller name="preprocessing.num_transform" control={form.control} render={({ field }) => (
                        <Select onValueChange={field.onChange} value={field.value}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">なし</SelectItem>
                            <SelectItem value="boxcox">Box-Cox 変換</SelectItem>
                            <SelectItem value="yeojohnson">Yeo-Johnson 変換</SelectItem>
                            <SelectItem value="quantile_uniform">Quantile (Uniform)</SelectItem>
                            <SelectItem value="quantile_normal">Quantile (Normal)</SelectItem>
                            <SelectItem value="log1p">log(1+x)</SelectItem>
                          </SelectContent>
                        </Select>
                      )} />
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-6 bg-emerald-500 rounded-full" />
                    <h3 className="font-bold text-slate-800">カテゴリ列 (Categorical)</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-1.5">
                      <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">エンコーディング</Label>
                      <Controller name="preprocessing.cat_encoder" control={form.control} render={({ field }) => (
                        <Select onValueChange={field.onChange} value={field.value}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="onehot">One-Hot Encoding</SelectItem>
                            <SelectItem value="ordinal">Ordinal Encoding</SelectItem>
                            <SelectItem value="target">Target Encoding</SelectItem>
                            <SelectItem value="binary">Binary Encoding</SelectItem>
                            <SelectItem value="leave_one_out">Leave-One-Out</SelectItem>
                          </SelectContent>
                        </Select>
                      )} />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">欠損値補完</Label>
                      <Controller name="preprocessing.cat_imputer" control={form.control} render={({ field }) => (
                        <Select onValueChange={field.onChange} value={field.value}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="most_frequent">最頻値</SelectItem>
                            <SelectItem value="constant">定数 ('missing')</SelectItem>
                            <SelectItem value="drop">行削除</SelectItem>
                          </SelectContent>
                        </Select>
                      )} />
                    </div>
                  </div>
                </div>
              </div>
              
              <Separator className="bg-slate-100" />
              
              <div className="space-y-4">
                <Label className="text-sm font-bold text-slate-700">除外する列（説明変数から除外）</Label>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mt-2">
                  {columns.filter(c => c.name !== targetCol).map(col => (
                    <label key={col.name} className="flex items-center gap-3 p-3 bg-white border border-slate-200 rounded-xl cursor-pointer hover:border-indigo-300 hover:bg-slate-50 transition-all group">
                      <input type="checkbox" value={col.name} {...form.register('preprocessing.exclude_cols')} className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500" />
                      <span className="text-sm font-medium text-slate-600 group-hover:text-slate-900 truncate">{col.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* --- Feature Generation Tab --- */}
        <TabsContent value="features">
          <Card className="border-none shadow-lg">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg">特徴量エンジニアリング & 外部プラグイン</CardTitle>
              <CardDescription>多項式特徴量や、SMILESベースの化学記述子エンジンを設定します。</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-10">
              <div className="p-6 bg-slate-900 rounded-2xl text-white space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold">Polynomial Features (多項式特徴量)</h3>
                  <Controller name="feature_generation.do_polynomial" control={form.control} render={({ field }) => (
                    <Switch checked={field.value} onCheckedChange={field.onChange} />
                  )} />
                </div>
                <div className={`grid md:grid-cols-2 gap-8 transition-opacity duration-300 ${form.watch('feature_generation.do_polynomial') ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
                  <div className="space-y-4">
                    <Label className="text-slate-400 text-xs font-bold uppercase">次数 (Degree)</Label>
                    <div className="flex items-center gap-6">
                      <Controller name="feature_generation.poly_degree" control={form.control} render={({ field }) => (
                        <Slider min={2} max={4} step={1} value={[field.value]} onValueChange={(val) => field.onChange(val[0])} className="flex-1" />
                      )} />
                      <span className="text-2xl font-black text-indigo-400 w-8">{form.watch('feature_generation.poly_degree')}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 pt-4">
                    <Controller name="feature_generation.poly_interaction_only" control={form.control} render={({ field }) => (
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    )} />
                    <div>
                      <Label className="block">交互作用のみ計算 (Interaction Only)</Label>
                      <p className="text-xs text-slate-500">x² 等の自己項を除去し、x * y のみを生成します。</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-bold text-slate-800">SMILES / 化学記述子エンジン</h3>
                  <div className="flex gap-2">
                    <Select value={selectedEngine} onValueChange={setSelectedEngine}>
                      <SelectTrigger className="w-64"><SelectValue placeholder="追加するエンジンを選択..." /></SelectTrigger>
                      <SelectContent>
                        {Object.entries(featureEngineSchemas).map(([key, val]: [string, any]) => (
                          <SelectItem key={key} value={key} disabled={engineFields.some(f => f.key === key)}>{val.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button onClick={addEngine} disabled={!selectedEngine} className="bg-slate-900"><Plus className="w-4 h-4 mr-2" />追加</Button>
                  </div>
                </div>

                <div className="grid gap-4">
                  {engineFields.map((field, index) => {
                    const schema = featureEngineSchemas[field.key];
                    return (
                      <Card key={field.id} className="border-slate-200 overflow-hidden group hover:border-indigo-300 transition-colors">
                        <div className="bg-slate-50 px-4 py-3 border-b flex justify-between items-center">
                          <div className="flex items-center gap-3">
                            <Controller name={`feature_generation.engines.${index}.enabled`} control={form.control} render={({ field }) => (
                              <Switch checked={field.value} onCheckedChange={field.onChange} />
                            )} />
                            <span className="font-bold text-slate-700">{schema?.name || field.key}</span>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => removeEngine(index)} className="text-slate-400 hover:text-red-500"><Trash2 className="w-4 h-4" /></Button>
                        </div>
                        {schema?.params && (
                          <CardContent className="p-5 grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {Object.entries(schema.params).map(([pKey, pConfig]: [string, any]) => (
                              <div key={pKey} className="space-y-1.5">
                                <Label className="text-xs font-bold text-slate-500">{pConfig.label || pKey}</Label>
                                {pConfig.type === 'toggle' || pConfig.type === 'boolean' ? (
                                  <div className="flex items-center gap-2">
                                    <Controller name={`feature_generation.engines.${index}.params.${pKey}`} control={form.control} render={({ field }) => (
                                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                                    )} />
                                    <span className="text-xs text-slate-600">{pConfig.description}</span>
                                  </div>
                                ) : pConfig.type === 'select' ? (
                                    <Controller name={`feature_generation.engines.${index}.params.${pKey}`} control={form.control} render={({ field }) => (
                                        <Select onValueChange={field.onChange} value={field.value}>
                                            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                            <SelectContent>
                                                {pConfig.options?.map((opt: any) => <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>)}
                                            </SelectContent>
                                        </Select>
                                    )} />
                                ) : (
                                  <Input 
                                    className="h-8 text-xs" 
                                    {...form.register(`feature_generation.engines.${index}.params.${pKey}` as any)} 
                                    defaultValue={pConfig.default} 
                                  />
                                )}
                              </div>
                            ))}
                          </CardContent>
                        )}
                      </Card>
                    );
                  })}
                  {engineFields.length === 0 && (
                    <div className="text-center py-12 border-2 border-dashed border-slate-200 rounded-2xl">
                      <Settings2 className="w-12 h-12 text-slate-200 mx-auto mb-3" />
                      <p className="text-slate-400">エンジンが追加されていません。SMILESデータがある場合はRDKit等を追加してください。</p>
                    </div>
                  )}
                </div>
              </div>

              <Separator />
              
              <div className="space-y-6">
                <h3 className="text-xl font-bold text-slate-800">Feature Selection (特徴量選択)</h3>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-1.5">
                    <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">アルゴリズム</Label>
                    <Controller name="feature_selection.feature_selector" control={form.control} render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">なし (すべて使用)</SelectItem>
                          <SelectItem value="variance">VarianceThreshold (低分散削除)</SelectItem>
                          <SelectItem value="selectkbest_f">SelectKBest (F-Score)</SelectItem>
                          <SelectItem value="select_from_model_lasso">Lasso 回帰による選択</SelectItem>
                          <SelectItem value="select_from_model_rf">RandomForest 重要度による選択</SelectItem>
                          <SelectItem value="rfe">RFE (再帰的特徴消去)</SelectItem>
                          <SelectItem value="boruta">Boruta SHAP (高度な選択)</SelectItem>
                        </SelectContent>
                      </Select>
                    )} />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">選択する特徴量数</Label>
                    <div className="flex items-center gap-4">
                        <Controller name="feature_selection.n_features_to_select" control={form.control} render={({ field }) => (
                            <Input type="number" min={1} {...field} className="flex-1" />
                        )} />
                        <span className="text-xs text-slate-400">Features</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* --- Model Tab --- */}
        <TabsContent value="model">
          <Card className="border-none shadow-lg">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg">学習アルゴリズム & ハイパーパラメータ</CardTitle>
              <CardDescription>タスクに最適な予測モデルを選択し、詳細なパラメータを設定します。</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-10">
              <div className="max-w-xl">
                <Label className="text-sm font-bold text-slate-700">Estimator (推定器)</Label>
                <Controller name="model.estimator" control={form.control} render={({ field }) => (
                  <Select onValueChange={field.onChange} value={field.value}>
                    <SelectTrigger className="mt-2 h-12 text-base font-semibold"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {availableEstimators?.map((est: any) => (
                        <SelectItem key={est.key} value={est.key}>{est.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )} />
              </div>

              {estimatorSchema && (
                <div className="space-y-6">
                    <div className="flex items-center gap-3">
                        <h3 className="text-xl font-bold text-slate-800">{estimatorSchema.name} の構成</h3>
                        <Badge variant="outline" className="bg-indigo-50 text-indigo-700 border-indigo-100">{estimatorSchema.key}</Badge>
                    </div>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {Object.entries(estimatorSchema.params).map(([pKey, pConfig]: [string, any]) => (
                        <div key={pKey} className="space-y-2 p-4 bg-slate-50 rounded-2xl border border-transparent hover:border-slate-200 transition-all">
                            <div className="flex justify-between">
                                <Label className="text-xs font-black uppercase text-slate-500">{pConfig.label || pKey}</Label>
                                {pConfig.type === 'slider' && <span className="text-xs font-mono font-bold text-indigo-600">{form.watch(`model.params.${pKey}`)}</span>}
                            </div>
                            <p className="text-[10px] text-slate-400 leading-tight mb-2">{pConfig.description}</p>
                            
                            {pConfig.type === 'toggle' ? (
                                <Controller name={`model.params.${pKey}`} control={form.control} render={({ field }) => (
                                    <Switch checked={field.value} onCheckedChange={field.onChange} />
                                )} />
                            ) : pConfig.type === 'slider' ? (
                                <Controller name={`model.params.${pKey}`} control={form.control} render={({ field }) => (
                                    <Slider min={pConfig.min} max={pConfig.max} step={pConfig.step} value={[field.value ?? pConfig.default]} onValueChange={(val) => field.onChange(val[0])} />
                                )} />
                            ) : pConfig.type === 'select' ? (
                                <Controller name={`model.params.${pKey}`} control={form.control} render={({ field }) => (
                                    <Select onValueChange={field.onChange} value={field.value}>
                                        <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
                                        <SelectContent>
                                            {pConfig.options?.map((opt: any) => <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>)}
                                        </SelectContent>
                                    </Select>
                                )} />
                            ) : (
                                <Input className="h-9 text-xs font-mono" {...form.register(`model.params.${pKey}` as any)} defaultValue={pConfig.default} />
                            )}
                        </div>
                        ))}
                    </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* --- Constraints Tab --- */}
        <TabsContent value="constraints">
          <Card className="border-none shadow-lg">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg">Monotonicity & Physical Constraints (物理・単調性制約)</CardTitle>
              <CardDescription>変数の増減に関するドメイン知識をモデルに強制または誘導します。</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-6">
              <div className="space-y-4">
                {form.watch('constraints').map((_, index) => (
                  <Card key={index} className="p-6 border-slate-200 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-orange-400" />
                    <div className="grid md:grid-cols-4 gap-6 items-end">
                      <div className="space-y-1.5">
                        <Label className="text-xs font-bold text-slate-500">対象変数</Label>
                        <Controller name={`constraints.${index}.feature`} control={form.control} render={({ field }) => (
                          <Select onValueChange={field.onChange} value={field.value}>
                            <SelectTrigger className="h-10"><SelectValue placeholder="変数を選択" /></SelectTrigger>
                            <SelectContent>{numericCols.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                          </Select>
                        )} />
                      </div>
                      <div className="space-y-1.5">
                        <Label className="text-xs font-bold text-slate-500">単調性方向</Label>
                        <Controller name={`constraints.${index}.direction`} control={form.control} render={({ field }) => (
                          <Select onValueChange={(val) => field.onChange(parseInt(val))} value={field.value?.toString()}>
                            <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="1">単調増加 ↗ (Positive)</SelectItem>
                              <SelectItem value="-1">単調減少 ↘ (Negative)</SelectItem>
                              <SelectItem value="0">単調 (方向不明)</SelectItem>
                            </SelectContent>
                          </Select>
                        )} />
                      </div>
                      <div className="space-y-1.5">
                        <Label className="text-xs font-bold text-slate-500">制約の強さ</Label>
                        <Controller name={`constraints.${index}.strength`} control={form.control} render={({ field }) => (
                          <Select onValueChange={field.onChange} value={field.value}>
                            <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="hard">Hard Constraint (絶対遵守)</SelectItem>
                              <SelectItem value="soft">Soft Constraint (ペナルティ)</SelectItem>
                            </SelectContent>
                          </Select>
                        )} />
                      </div>
                      <div className="flex justify-end pb-1">
                        <Button variant="ghost" size="icon" onClick={() => { const current = form.getValues('constraints'); current.splice(index, 1); form.setValue('constraints', current); }} className="text-slate-300 hover:text-red-500">
                          <Trash2 className="w-5 h-5" />
                        </Button>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-10 mt-6 pt-6 border-t border-slate-100">
                      <div className="space-y-3">
                        <div className="flex justify-between">
                            <Label className="text-xs font-bold text-slate-400">適用範囲 (±nσ Sigma Range)</Label>
                            <span className="text-sm font-bold text-orange-600">{form.watch(`constraints.${index}.sigma_range`)}σ</span>
                        </div>
                        <Controller name={`constraints.${index}.sigma_range`} control={form.control} render={({ field }) => (
                          <Slider min={0.5} max={10.0} step={0.5} value={[field.value]} onValueChange={(val) => field.onChange(val[0])} />
                        )} />
                        <p className="text-[10px] text-slate-400">平均からどれだけ離れたデータにまで制約を適用するかを制御します。</p>
                      </div>
                      <div className="flex items-center gap-4 pt-4">
                        <Controller name={`constraints.${index}.linear`} control={form.control} render={({ field }) => (
                          <Switch checked={field.value} onCheckedChange={field.onChange} />
                        )} />
                        <div>
                            <Label className="font-bold text-slate-700">線形性制約も適用</Label>
                            <p className="text-[10px] text-slate-400">単調性だけでなく、振る舞いを線形に近づけます。</p>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
                
                <Button 
                    variant="outline" 
                    className="w-full h-16 border-dashed border-2 border-slate-200 hover:border-orange-300 hover:bg-orange-50 transition-all text-slate-400 hover:text-orange-600" 
                    onClick={() => { const current = form.getValues('constraints'); current.push({ feature: numericCols[0] || '', direction: 1, strength: 'soft', sigma_range: 3.0, linear: false }); form.setValue('constraints', current); }}
                >
                  <Plus className="w-5 h-5 mr-2" />物理・ドメイン制約を追加
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* --- Execution Tab --- */}
        <TabsContent value="execution">
          <Card className="border-none shadow-lg">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="text-lg">バリデーション & 解析実行設定</CardTitle>
              <CardDescription>モデルの信頼性を評価するためのバリデーション戦略を決定します。</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-10">
              <div className="grid md:grid-cols-2 gap-10">
                <div className="space-y-6">
                  <div className="space-y-1.5">
                    <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">交差検証戦略 (CV Strategy)</Label>
                    <Controller name="execution.cv_strategy" control={form.control} render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger className="h-12"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="kfold">K-Fold Cross Validation</SelectItem>
                          <SelectItem value="stratified">Stratified K-Fold (層化)</SelectItem>
                          <SelectItem value="group">Group K-Fold (外部データ分離)</SelectItem>
                          <SelectItem value="time_series">Time Series Split (時系列)</SelectItem>
                          <SelectItem value="shuffle">Shuffle Split</SelectItem>
                        </SelectContent>
                      </Select>
                    )} />
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                        <Label className="text-xs font-bold uppercase text-slate-400 tracking-wider">分割数 (Folds)</Label>
                        <span className="text-lg font-black text-indigo-600">{form.watch('execution.cv_folds')} Folds</span>
                    </div>
                    <Controller name="execution.cv_folds" control={form.control} render={({ field }) => (
                      <Slider min={2} max={10} step={1} value={[field.value]} onValueChange={(val) => field.onChange(val[0])} />
                    )} />
                  </div>
                </div>

                <div className="space-y-6">
                  <h3 className="font-bold text-slate-800 mb-4">事後解析・出力オプション</h3>
                  <div className="grid gap-4">
                    <label className="flex items-center gap-4 p-4 bg-white border border-slate-200 rounded-2xl cursor-pointer hover:border-indigo-300 transition-all">
                        <Controller name="execution.do_eda" control={form.control} render={({ field }) => (
                            <Switch checked={field.value} onCheckedChange={field.onChange} />
                        )} />
                        <div>
                            <span className="block font-bold text-slate-700">自動 EDA 実行</span>
                            <span className="text-[10px] text-slate-400">学習データセットの統計分布を自動記録します。</span>
                        </div>
                    </label>
                    <label className="flex items-center gap-4 p-4 bg-white border border-slate-200 rounded-2xl cursor-pointer hover:border-indigo-300 transition-all">
                        <Controller name="execution.do_shap" control={form.control} render={({ field }) => (
                            <Switch checked={field.value} onCheckedChange={field.onChange} />
                        )} />
                        <div>
                            <span className="block font-bold text-slate-700">SHAP 値の計算 (解釈可能性)</span>
                            <span className="text-[10px] text-slate-400">特徴量の寄与をサンプル単位で詳細に分析します。</span>
                        </div>
                    </label>
                  </div>
                </div>
              </div>

              <div className="p-8 bg-indigo-600 rounded-3xl text-white flex flex-col md:flex-row items-center justify-between gap-6 shadow-xl shadow-indigo-200">
                <div className="space-y-2">
                    <h3 className="text-2xl font-black">Ready to launch?</h3>
                    <p className="text-indigo-100 text-sm opacity-80">設定が完了しました。解析を実行してモデルを構築しましょう。</p>
                </div>
                <Button 
                    size="lg" 
                    onClick={() => runMutation.mutate(form.getValues())} 
                    disabled={runMutation.isPending || !targetCol}
                    className="bg-white text-indigo-600 hover:bg-slate-100 h-14 px-10 text-lg font-black rounded-2xl shadow-lg active:scale-95 transition-all"
                >
                    {runMutation.isPending ? <Loader2 className="w-6 h-6 animate-spin mr-3" /> : <Play className="w-6 h-6 mr-3 fill-current" />}
                    解析エンジンを起動
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
