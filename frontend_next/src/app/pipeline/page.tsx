// frontend_next/src/app/pipeline/page.tsx
'use client';

import { useState, useEffect, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useChemAIStore } from '@/lib/store';
import { getPipelineConfig, updatePipelineConfig, runPipeline, getAvailableModels, getAvailableFeatureEngines } from '@/lib/api';
import { getEstimatorSchema, getEstimatorsForTask } from '@/config/estimatorSchemas';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '@/components/ui/accordion';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Loader2, Settings, Play, Save, RefreshCw, AlertCircle, Plus, Trash2, Info } from 'lucide-react';
import { useRouter } from 'next/navigation';

// Form schema for pipeline configuration
const pipelineSchema = z.object({
  cv_strategy: z.enum(['kfold', 'stratified', 'group', 'time_series', 'loo', 'lgo']),
  cv_folds: z.number().min(2).max(10),
  preprocessing: z.object({
    num_scaler: z.enum(['standard', 'robust', 'minmax', 'maxabs', 'none']),
    num_imputer: z.enum(['median', 'mean', 'knn', 'iterative', 'drop']),
    num_transform: z.enum(['none', 'boxcox', 'yeojohnson', 'quantile_uniform', 'quantile_normal', 'log1p']),
    cat_encoder: z.enum(['onehot', 'ordinal', 'target', 'binary', 'leave_one_out']),
    cat_imputer: z.enum(['most_frequent', 'constant', 'drop']),
  }),
  feature_generation: z.object({
    do_polynomial: z.boolean(),
    poly_degree: z.number().min(2).max(3),
    poly_interaction_only: z.boolean(),
  }),
  feature_selection: z.object({
    feature_selector: z.enum(['none', 'variance', 'selectkbest_f', 'selectkbest_mi', 'select_from_model_lasso', 'select_from_model_rf', 'rfe', 'boruta']),
    n_features_to_select: z.number().min(1).max(1000),
  }),
  estimator: z.string().min(1),
  estimator_params: z.record(z.any()),
  monotonic_constraints: z.array(z.object({
    feature: z.string(),
    direction: z.enum([-1, 0, 1] as const),
    strength: z.enum(['hard', 'soft']),
    sigma_range: z.number(),
    linear: z.boolean(),
  })),
  do_eda: z.boolean(),
  do_prep: z.boolean(),
  do_eval: z.boolean(),
  do_pca: z.boolean(),
  do_shap: z.boolean(),
});

type PipelineFormValues = z.infer<typeof pipelineSchema>;

export default function PipelinePage() {
  const router = useRouter();
  const { pipelineConfig, updatePipelineConfig: updateStoreConfig, targetCol, taskType, columns, setError, setAnalysisResult } = useChemAIStore();
  const [activeTab, setActiveTab] = useState('preprocessing');

  // Fetch available models
  const { data: availableEstimators } = useQuery({
    queryKey: ['estimators', taskType],
    queryFn: () => getAvailableModels(taskType),
    enabled: !!taskType,
  });

  // React Hook Form setup
  const form = useForm<PipelineFormValues>({
    resolver: zodResolver(pipelineSchema),
    defaultValues: pipelineConfig as PipelineFormValues,
  });

  // Sync form with store when config changes externally
  useEffect(() => {
    form.reset(pipelineConfig);
  }, [pipelineConfig, form]);

  const selectedEstimator = form.watch('estimator');
  const estimatorSchema = useMemo(() => {
    if (!selectedEstimator || !taskType) return null;
    return getEstimatorSchema(selectedEstimator, taskType);
  }, [selectedEstimator, taskType]);

  // Actions
  const onSave = (data: PipelineFormValues) => {
    updateStoreConfig(data);
  };

  const runPipelineMutation = useMutation({
    mutationFn: (config: PipelineConfig) => runPipeline(config),
    onSuccess: (result) => {
      setAnalysisResult(result);
      router.push('/results');
    },
    onError: (error: any) => {
      setError(`解析実行に失敗: ${error.message}`);
    },
  });

  const onRun = () => {
    const data = form.getValues();
    updateStoreConfig(data);
    runPipelineMutation.mutate(data);
  };

  const renderParamField = (paramName: string, config: any) => {
    const path = `estimator_params.${paramName}` as const;
    switch (config.type) {
      case 'number':
        return (
          <div key={paramName} className="space-y-2">
            <Label className="text-xs font-bold uppercase text-slate-500">{config.label}</Label>
            <Controller
              name={path}
              control={form.control}
              render={({ field }) => (
                <Input
                  type="number"
                  {...field}
                  onChange={(e) => field.onChange(Number(e.target.value))}
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  className="h-9"
                />
              )}
            />
            <p className="text-[10px] text-slate-400">{config.description}</p>
          </div>
        );
      case 'slider':
        return (
          <div key={paramName} className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-bold uppercase text-slate-500">{config.label}</Label>
              <Controller
                name={path}
                control={form.control}
                render={({ field }) => (
                  <span className="text-xs font-mono bg-slate-100 px-2 py-0.5 rounded">{field.value}</span>
                )}
              />
            </div>
            <Controller
              name={path}
              control={form.control}
              render={({ field }) => (
                <Slider
                  value={[field.value]}
                  onValueChange={(vals) => field.onChange(vals[0])}
                  min={config.min}
                  max={config.max}
                  step={config.step}
                />
              )}
            />
            <p className="text-[10px] text-slate-400">{config.description}</p>
          </div>
        );
      case 'toggle':
        return (
          <div key={paramName} className="flex items-center justify-between p-3 rounded-lg border border-slate-100">
            <div className="space-y-1">
              <Label className="text-sm font-bold text-slate-700">{config.label}</Label>
              <p className="text-[10px] text-slate-400">{config.description}</p>
            </div>
            <Controller
              name={path}
              control={form.control}
              render={({ field }) => (
                <Switch checked={field.value} onCheckedChange={field.onChange} />
              )}
            />
          </div>
        );
      case 'select':
        return (
          <div key={paramName} className="space-y-2">
            <Label className="text-xs font-bold uppercase text-slate-500">{config.label}</Label>
            <Controller
              name={path}
              control={form.control}
              render={({ field }) => (
                <Select onValueChange={field.onChange} value={field.value}>
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {config.options?.map((opt: any) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
            <p className="text-[10px] text-slate-400">{config.description}</p>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Pipeline Configuration</h1>
          <p className="text-slate-500 mt-1">Design your machine learning workflow with precision.</p>
        </div>
        <div className="flex items-center gap-3">
          <Button 
            variant="outline" 
            onClick={() => form.reset(pipelineConfig)}
            className="font-bold"
          >
            <RefreshCw className="w-4 h-4 mr-2" /> Reset
          </Button>
          <Button 
            onClick={form.handleSubmit(onSave)}
            variant="secondary"
            className="font-bold"
          >
            <Save className="w-4 h-4 mr-2" /> Save Config
          </Button>
          <Button 
            onClick={onRun}
            disabled={runPipelineMutation.isPending || !targetCol}
            className="bg-indigo-600 hover:bg-indigo-700 font-bold px-8 shadow-lg shadow-indigo-100"
          >
            {runPipelineMutation.isPending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Play className="w-4 h-4 mr-2" />
            )}
            Run Pipeline
          </Button>
        </div>
      </div>

      {!targetCol && (
        <div className="mb-8 p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-center gap-3 text-amber-800">
          <Info className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm font-medium">
            解析を実行するには「Data」タブでデータをロードしてください。
          </p>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="bg-slate-100/50 p-1 rounded-2xl w-full max-w-3xl border border-slate-200">
          <TabsTrigger value="preprocessing" className="rounded-xl font-bold py-2 px-6">Preprocessing</TabsTrigger>
          <TabsTrigger value="features" className="rounded-xl font-bold py-2 px-6">Features</TabsTrigger>
          <TabsTrigger value="model" className="rounded-xl font-bold py-2 px-6">Model</TabsTrigger>
          <TabsTrigger value="constraints" className="rounded-xl font-bold py-2 px-6">Constraints</TabsTrigger>
          <TabsTrigger value="advanced" className="rounded-xl font-bold py-2 px-6">Execution</TabsTrigger>
        </TabsList>

        <TabsContent value="preprocessing" className="animate-in fade-in slide-in-from-left-4 duration-300">
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Numeric Pipeline</CardTitle>
                <CardDescription>Handling continuous and discrete values.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Scaler</Label>
                  <Controller
                    name="preprocessing.num_scaler"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="standard">StandardScaler</SelectItem>
                          <SelectItem value="robust">RobustScaler</SelectItem>
                          <SelectItem value="minmax">MinMaxScaler</SelectItem>
                          <SelectItem value="maxabs">MaxAbsScaler</SelectItem>
                          <SelectItem value="none">None</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Imputer</Label>
                  <Controller
                    name="preprocessing.num_imputer"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="median">Median</SelectItem>
                          <SelectItem value="mean">Mean</SelectItem>
                          <SelectItem value="knn">KNN Imputer</SelectItem>
                          <SelectItem value="iterative">Iterative Imputer</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Non-linear Transform</Label>
                  <Controller
                    name="preprocessing.num_transform"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="boxcox">Box-Cox</SelectItem>
                          <SelectItem value="yeojohnson">Yeo-Johnson</SelectItem>
                          <SelectItem value="log1p">Log(1+x)</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Categorical Pipeline</CardTitle>
                <CardDescription>Encoding and missing values for strings.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Encoder</Label>
                  <Controller
                    name="preprocessing.cat_encoder"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="onehot">OneHotEncoder</SelectItem>
                          <SelectItem value="ordinal">OrdinalEncoder</SelectItem>
                          <SelectItem value="target">TargetEncoder</SelectItem>
                          <SelectItem value="binary">BinaryEncoder</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Imputer</Label>
                  <Controller
                    name="preprocessing.cat_imputer"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="most_frequent">Most Frequent</SelectItem>
                          <SelectItem value="constant">Constant ('missing')</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="features" className="animate-in fade-in slide-in-from-left-4 duration-300">
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Polynomial Features</CardTitle>
                <CardDescription>Generate interaction terms.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <Label>Enable Polynomial</Label>
                  <Controller
                    name="feature_generation.do_polynomial"
                    control={form.control}
                    render={({ field }) => (
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    )}
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label>Degree</Label>
                    <Controller
                      name="feature_generation.poly_degree"
                      control={form.control}
                      render={({ field }) => <span className="font-bold text-indigo-600">{field.value}</span>}
                    />
                  </div>
                  <Controller
                    name="feature_generation.poly_degree"
                    control={form.control}
                    render={({ field }) => (
                      <Slider
                        value={[field.value]}
                        onValueChange={(vals) => field.onChange(vals[0])}
                        min={2}
                        max={3}
                        step={1}
                        disabled={!form.watch('feature_generation.do_polynomial')}
                      />
                    )}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Feature Selection</CardTitle>
                <CardDescription>Reduce dimensionality automatically.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Method</Label>
                  <Controller
                    name="feature_selection.feature_selector"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="variance">VarianceThreshold</SelectItem>
                          <SelectItem value="selectkbest_mi">SelectKBest (MI)</SelectItem>
                          <SelectItem value="select_from_model_rf">SelectFromModel (RF)</SelectItem>
                          <SelectItem value="rfe">RFECV</SelectItem>
                          <SelectItem value="boruta">Boruta</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Number of Features to Select</Label>
                  <Controller
                    name="feature_selection.n_features_to_select"
                    control={form.control}
                    render={({ field }) => (
                      <Input
                        type="number"
                        {...field}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                        disabled={form.watch('feature_selection.feature_selector') === 'none'}
                      />
                    )}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="model" className="animate-in fade-in slide-in-from-left-4 duration-300">
          <div className="grid lg:grid-cols-3 gap-8">
            <Card className="lg:col-span-1 border-slate-200 shadow-sm h-fit">
              <CardHeader>
                <CardTitle className="text-lg">Algorithm</CardTitle>
                <CardDescription>Select base estimator.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Model Class</Label>
                  <Controller
                    name="estimator"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger className="h-12 border-2 border-indigo-100 hover:border-indigo-200">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {getEstimatorsForTask(taskType).map(est => (
                            <SelectItem key={est.key} value={est.key}>{est.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                {estimatorSchema && (
                  <div className="p-3 bg-indigo-50 rounded-xl text-[10px] text-indigo-700 leading-relaxed font-medium">
                    <p className="font-bold mb-1">About {estimatorSchema.name}</p>
                    {estimatorSchema.description}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="lg:col-span-2 border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Hyperparameters</CardTitle>
                <CardDescription>Configure specific parameters for {selectedEstimator}.</CardDescription>
              </CardHeader>
              <CardContent>
                {estimatorSchema ? (
                  <div className="grid md:grid-cols-2 gap-x-8 gap-y-6">
                    {Object.entries(estimatorSchema.params).map(([key, config]) => 
                      renderParamField(key, config)
                    )}
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-400">
                    <Settings className="mx-auto w-12 h-12 mb-4 opacity-20" />
                    <p>Select a model to configure parameters.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="constraints" className="animate-in fade-in slide-in-from-left-4 duration-300">
          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg">Monotonic Constraints</CardTitle>
                <CardDescription>Enforce logical domain knowledge (e.g., concentration increases activity).</CardDescription>
              </div>
              <Button 
                size="sm" 
                variant="outline" 
                onClick={() => {
                  const current = form.getValues('monotonic_constraints');
                  form.setValue('monotonic_constraints', [
                    ...current,
                    { feature: columns[0]?.name || '', direction: 1, strength: 'hard', sigma_range: 3, linear: false }
                  ]);
                }}
              >
                <Plus className="w-4 h-4 mr-2" /> Add Constraint
              </Button>
            </CardHeader>
            <CardContent>
              {form.watch('monotonic_constraints').length > 0 ? (
                <div className="space-y-4">
                  {form.watch('monotonic_constraints').map((constraint, index) => (
                    <div key={index} className="grid md:grid-cols-5 gap-4 p-4 bg-slate-50 rounded-2xl border border-slate-100 items-end">
                      <div className="space-y-2">
                        <Label className="text-xs">Feature</Label>
                        <Controller
                          name={`monotonic_constraints.${index}.feature`}
                          control={form.control}
                          render={({ field }) => (
                            <Select onValueChange={field.onChange} value={field.value}>
                              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                              <SelectContent>
                                {columns.map(c => <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>)}
                              </SelectContent>
                            </Select>
                          )}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Direction</Label>
                        <Controller
                          name={`monotonic_constraints.${index}.direction`}
                          control={form.control}
                          render={({ field }) => (
                            <Select 
                              onValueChange={(val) => field.onChange(parseInt(val))} 
                              value={field.value.toString()}
                            >
                              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="1">Increasing (+)</SelectItem>
                                <SelectItem value="-1">Decreasing (-)</SelectItem>
                              </SelectContent>
                            </Select>
                          )}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Strength</Label>
                        <Controller
                          name={`monotonic_constraints.${index}.strength`}
                          control={form.control}
                          render={({ field }) => (
                            <Select onValueChange={field.onChange} value={field.value}>
                              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="hard">Hard Constraint</SelectItem>
                                <SelectItem value="soft">Soft Constraint</SelectItem>
                              </SelectContent>
                            </Select>
                          )}
                        />
                      </div>
                      <div className="flex items-center gap-4 h-9">
                        <div className="flex items-center gap-2">
                          <Label className="text-xs">Linear</Label>
                          <Controller
                            name={`monotonic_constraints.${index}.linear`}
                            control={form.control}
                            render={({ field }) => (
                              <Switch checked={field.value} onCheckedChange={field.onChange} />
                            )}
                          />
                        </div>
                        <Button 
                          size="icon" 
                          variant="ghost" 
                          className="text-red-500 hover:text-red-700 hover:bg-red-50 ml-auto"
                          onClick={() => {
                            const current = form.getValues('monotonic_constraints');
                            form.setValue('monotonic_constraints', current.filter((_, i) => i !== index));
                          }}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-12 text-center text-slate-400 border-2 border-dashed border-slate-100 rounded-2xl">
                  <p>No constraints defined yet.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced" className="animate-in fade-in slide-in-from-left-4 duration-300">
           <div className="grid md:grid-cols-2 gap-8">
            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Cross-Validation</CardTitle>
                <CardDescription>Robustness evaluation strategy.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Strategy</Label>
                  <Controller
                    name="cv_strategy"
                    control={form.control}
                    render={({ field }) => (
                      <Select onValueChange={field.onChange} value={field.value}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="kfold">K-Fold</SelectItem>
                          <SelectItem value="stratified">Stratified K-Fold</SelectItem>
                          <SelectItem value="group">Group K-Fold</SelectItem>
                          <SelectItem value="time_series">Time Series Split</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>Folds</Label>
                    <Controller
                      name="cv_folds"
                      control={form.control}
                      render={({ field }) => <span className="font-bold text-indigo-600">{field.value}</span>}
                    />
                  </div>
                  <Controller
                    name="cv_folds"
                    control={form.control}
                    render={({ field }) => (
                      <Slider
                        value={[field.value]}
                        onValueChange={(vals) => field.onChange(vals[0])}
                        min={2}
                        max={10}
                        step={1}
                      />
                    )}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Analysis Flags</CardTitle>
                <CardDescription>Enable optional compute-heavy modules.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {[
                  { key: 'do_eda', label: 'EDA (Statistics & Plotting)' },
                  { key: 'do_pca', label: 'Dimensionality Reduction (PCA)' },
                  { key: 'do_shap', label: 'Model Interpretability (SHAP)' },
                  { key: 'do_eval', label: 'Detailed Validation Metrics' },
                ].map((flag) => (
                  <div key={flag.key} className="flex items-center justify-between p-3 hover:bg-slate-50 rounded-lg transition-colors">
                    <Label className="text-sm font-medium">{flag.label}</Label>
                    <Controller
                      name={flag.key as any}
                      control={form.control}
                      render={({ field }) => (
                        <Switch checked={field.value} onCheckedChange={field.onChange} />
                      )}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
