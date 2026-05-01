// frontend_next/src/components/ml/DynamicParamForm.tsx
// Dynamic form renderer based on backend-generated UI metadata

'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, Info, AlertCircle } from 'lucide-react';
import { UIParamMetadata, UIInputType } from '@/types/ml';

interface DynamicParamFormProps {
  params: Record<string, UIParamMetadata>;
  initialValues?: Record<string, any>;
  onChange: (paramName: string, value: any) => void;
  onValidate?: (paramName: string, value: any) => { valid: boolean; error?: string };
  showAdvanced?: boolean;
  className?: string;
}

export function DynamicParamForm({
  params,
  initialValues = {},
  onChange,
  onValidate,
  showAdvanced = false,
  className = ''
}: DynamicParamFormProps) {
  const [localValues, setLocalValues] = useState<Record<string, any>>(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['general']));

  // Initialize values from props
  useEffect(() => {
    setLocalValues(prev => ({ ...prev, ...initialValues }));
  }, [initialValues]);

  // Group parameters by category
  const groupedParams = useMemo(() => {
    const groups: Record<string, Record<string, UIParamMetadata>> = {};
    
    for (const [paramName, paramMeta] of Object.entries(params)) {
      if (paramName.startsWith('_')) continue; // Skip internal metadata
      
      const category = paramMeta.category || 'general';
      if (!groups[category]) {
        groups[category] = {};
      }
      groups[category][paramName] = paramMeta;
    }
    
    // Sort categories: general first, then alphabetical
    return Object.entries(groups)
      .sort(([a], [b]) => {
        if (a === 'general') return -1;
        if (b === 'general') return 1;
        return a.localeCompare(b);
      })
      .reduce((acc, [category, params]) => {
        acc[category] = params;
        return acc;
      }, {} as Record<string, Record<string, UIParamMetadata>>);
  }, [params]);

  // Handle value change with validation
  const handleChange = useCallback((paramName: string, value: any) => {
    setLocalValues(prev => ({ ...prev, [paramName]: value }));
    
    // Run validation if provided
    if (onValidate) {
      const result = onValidate(paramName, value);
      setErrors(prev => ({
        ...prev,
        [paramName]: result.valid ? undefined : result.error
      }));
    }
    
    // Notify parent
    onChange(paramName, value);
  }, [onChange, onValidate]);

  // Toggle category expansion
  const toggleCategory = useCallback((category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  // Render input based on type
  const renderInput = useCallback((paramName: string, paramMeta: UIParamMetadata) => {
    const value = localValues[paramName] ?? paramMeta.default;
    const error = errors[paramName];
    const disabled = paramMeta.disabled || false;
    
    const commonProps = {
      id: paramName,
      disabled,
      'aria-describedby': paramMeta.description ? `${paramName}-desc` : undefined,
    };

    switch (paramMeta.input_type) {
      case UIInputType.BOOLEAN:
        return (
          <div className="flex items-center space-x-2">
            <Switch
              {...commonProps}
              checked={!!value}
              onCheckedChange={(checked) => handleChange(paramName, checked)}
            />
            <Label htmlFor={paramName} className="cursor-pointer">
              {paramMeta.label || paramName}
            </Label>
          </div>
        );

      case UIInputType.SELECT:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
            <Select
              value={value?.toString() ?? ''}
              onValueChange={(val) => {
                const parsed = paramMeta.options?.find(o => o.value?.toString() === val)?.value ?? val;
                handleChange(paramName, parsed);
              }}
              disabled={disabled}
            >
              <SelectTrigger id={paramName}>
                <SelectValue placeholder={paramMeta.placeholder || "Select..."} />
              </SelectTrigger>
              <SelectContent>
                {paramMeta.options?.map((opt) => (
                  <SelectItem 
                    key={opt.value?.toString()} 
                    value={opt.value?.toString() ?? ''}
                    title={opt.description}
                  >
                    {opt.label || opt.value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );

      case UIInputType.MULTI_SELECT:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
            <Select
              value={Array.isArray(value) ? value[0]?.toString() : ''}
              onValueChange={(val) => {
                const parsed = paramMeta.options?.find(o => o.value?.toString() === val)?.value ?? val;
                const current = Array.isArray(localValues[paramName]) ? localValues[paramName] : [];
                const next = current.includes(parsed) 
                  ? current.filter((v: any) => v !== parsed)
                  : [...current, parsed];
                handleChange(paramName, next);
              }}
              disabled={disabled}
            >
              <SelectTrigger id={paramName}>
                <SelectValue placeholder={paramMeta.placeholder || "Select..."} />
              </SelectTrigger>
              <SelectContent>
                {paramMeta.options?.map((opt) => (
                  <SelectItem 
                    key={opt.value?.toString()} 
                    value={opt.value?.toString() ?? ''}
                  >
                    {opt.label || opt.value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {Array.isArray(value) && value.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {value.map((v: any, i: number) => (
                  <Badge key={i} variant="secondary" className="text-xs">
                    {v?.toString()}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        );

      case UIInputType.SLIDER:
        const min = paramMeta.min_value ?? 0;
        const max = paramMeta.max_value ?? 100;
        const step = paramMeta.step ?? 1;
        const numericValue = typeof value === 'number' ? value : (paramMeta.default ?? min);
        
        return (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
              <span className="text-sm font-mono text-muted-foreground">
                {typeof numericValue === 'number' ? numericValue.toFixed(step < 1 ? 2 : 0) : numericValue}
              </span>
            </div>
            <Slider
              {...commonProps}
              value={[numericValue]}
              min={min}
              max={max}
              step={step}
              onValueChange={([val]) => handleChange(paramName, val)}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{min}</span>
              <span>{max}</span>
            </div>
          </div>
        );

      case UIInputType.INTEGER:
      case UIInputType.NUMBER:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
            <Input
              {...commonProps}
              type={paramMeta.input_type === UIInputType.INTEGER ? 'number' : 'number'}
              value={value ?? ''}
              onChange={(e) => {
                const val = e.target.value;
                const parsed = paramMeta.input_type === UIInputType.INTEGER 
                  ? (val === '' ? undefined : parseInt(val, 10))
                  : (val === '' ? undefined : parseFloat(val));
                handleChange(paramName, isNaN(parsed as number) ? val : parsed);
              }}
              min={paramMeta.min_value}
              max={paramMeta.max_value}
              step={paramMeta.step}
              placeholder={paramMeta.placeholder}
              className={error ? 'border-destructive' : ''}
            />
            {paramMeta.min_value !== undefined && paramMeta.max_value !== undefined && (
              <p className="text-xs text-muted-foreground">
                Range: {paramMeta.min_value} – {paramMeta.max_value}
              </p>
            )}
          </div>
        );

      case UIInputType.TEXTAREA:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
            <Textarea
              {...commonProps}
              value={Array.isArray(value) ? value.join(', ') : (value?.toString() ?? '')}
              onChange={(e) => {
                const val = e.target.value;
                // Auto-parse comma-separated for list types
                if (paramMeta.input_type === UIInputType.MULTI_SELECT || paramMeta.name.includes('list')) {
                  const items = val.split(',').map((s: string) => {
                    const trimmed = s.trim();
                    const num = Number(trimmed);
                    return isNaN(num) ? trimmed : num;
                  }).filter((x: any) => x !== '');
                  handleChange(paramName, items);
                } else {
                  handleChange(paramName, val);
                }
              }}
              placeholder={paramMeta.placeholder}
              rows={3}
              className={error ? 'border-destructive' : ''}
            />
          </div>
        );

      default:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramMeta.label || paramName}</Label>
            <Input
              {...commonProps}
              type="text"
              value={value?.toString() ?? ''}
              onChange={(e) => handleChange(paramName, e.target.value)}
              placeholder={paramMeta.placeholder}
              pattern={paramMeta.pattern || undefined}
              minLength={paramMeta.min_length || undefined}
              maxLength={paramMeta.max_length || undefined}
              className={error ? 'border-destructive' : ''}
            />
          </div>
        );
    }
  }, [localValues, errors, handleChange]);

  // Check if parameter should be shown
  const shouldShowParam = useCallback((paramMeta: UIParamMetadata): boolean => {
    if (paramMeta.hidden) return false;
    if (paramMeta.advanced && !showAdvanced) return false;
    
    // Evaluate conditional display logic
    if (paramMeta.depends_on) {
      for (const [depParam, expectedValue] of Object.entries(paramMeta.depends_on)) {
        const actualValue = localValues[depParam];
        if (actualValue !== expectedValue) {
          return false;
        }
      }
    }
    
    return true;
  }, [localValues, showAdvanced]);

  return (
    <TooltipProvider>
      <div className={`space-y-4 ${className}`}>
        {/* Class/Function info header */}
        {params['_class_info'] && (
          <Card className="bg-muted/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                {(params['_class_info'] as any).name}
                <Badge variant="outline" className="text-xs">
                  {(params['_class_info'] as any).category}
                </Badge>
              </CardTitle>
              {(params['_class_info'] as any).description && (
                <p className="text-sm text-muted-foreground">
                  {(params['_class_info'] as any).description}
                </p>
              )}
            </CardHeader>
          </Card>
        )}

        {/* Parameter groups */}
        {Object.entries(groupedParams).map(([category, categoryParams]) => {
          const visibleParams = Object.entries(categoryParams)
            .filter(([_, meta]) => shouldShowParam(meta));
          
          if (visibleParams.length === 0) return null;
          
          const isExpanded = expandedCategories.has(category);
          const isGeneral = category === 'general';
          
          return (
            <Collapsible 
              key={category} 
              open={isExpanded || isGeneral}
              onOpenChange={() => toggleCategory(category)}
              className="border rounded-lg"
            >
              <CollapsibleTrigger className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors">
                <div className="flex items-center gap-2">
                  <ChevronDown className={`w-4 h-4 transition-transform ${isExpanded || isGeneral ? 'rotate-0' : '-rotate-90'}`} />
                  <span className="font-medium text-sm uppercase tracking-wide">
                    {category.replace('_', ' ')}
                  </span>
                  <Badge variant="secondary" className="text-xs">
                    {visibleParams.length}
                  </Badge>
                </div>
                {!isGeneral && (
                  <span className="text-xs text-muted-foreground">
                    {isExpanded ? 'Click to collapse' : 'Click to expand'}
                  </span>
                )}
              </CollapsibleTrigger>
              
              <CollapsibleContent>
                <CardContent className="pt-2 pb-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {visibleParams
                      .sort(([, a], [, b]) => (a.order || 0) - (b.order || 0))
                      .map(([paramName, paramMeta]) => (
                        <div key={paramName} className="space-y-1">
                          {renderInput(paramName, paramMeta)}
                          
                          {/* Description tooltip */}
                          {paramMeta.description && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button 
                                  id={`${paramName}-desc`}
                                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                                  type="button"
                                >
                                  <Info className="w-3 h-3" />
                                  <span>Details</span>
                                </button>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                <p className="text-xs">{paramMeta.description}</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                          
                          {/* Validation error */}
                          {errors[paramName] && (
                            <div className="flex items-center gap-1 text-xs text-destructive">
                              <AlertCircle className="w-3 h-3" />
                              <span>{errors[paramName]}</span>
                            </div>
                          )}
                        </div>
                      ))}
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
