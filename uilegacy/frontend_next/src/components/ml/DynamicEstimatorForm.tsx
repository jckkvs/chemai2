// frontend_next/src/components/ml/DynamicEstimatorForm.tsx
// Dynamic form generator for estimator parameters

import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { Info } from 'lucide-react';
import { EstimatorUIMetadata } from '@/types/ml';

interface DynamicEstimatorFormProps {
  estimatorName: string;
  metadata: EstimatorUIMetadata;
  initialValues?: Record<string, any>;
  onChange: (paramName: string, value: any) => void;
}

export function DynamicEstimatorForm({ 
  estimatorName, 
  metadata, 
  initialValues = {}, 
  onChange 
}: DynamicEstimatorFormProps) {
  
  // Group parameters by category (inferred from name patterns)
  const groupedParams = useMemo(() => {
    const groups: Record<string, EstimatorUIMetadata> = {
      'basic': {},
      'regularization': {},
      'optimization': {},
      'advanced': {}
    };
    
    for (const [paramName, paramMeta] of Object.entries(metadata)) {
      if (['alpha', 'lambda', 'reg_alpha', 'reg_lambda'].includes(paramName)) {
        groups['regularization'][paramName] = paramMeta;
      } else if (['learning_rate', 'n_estimators', 'max_iter'].includes(paramName)) {
        groups['optimization'][paramName] = paramMeta;
      } else if (paramMeta.description?.toLowerCase().includes('advanced') || paramName.startsWith('_')) {
        groups['advanced'][paramName] = paramMeta;
      } else {
        groups['basic'][paramName] = paramMeta;
      }
    }
    
    // Remove empty groups
    return Object.fromEntries(
      Object.entries(groups).filter(([_, params]) => Object.keys(params).length > 0)
    );
  }, [metadata]);

  const renderInput = (paramName: string, paramMeta: any) => {
    const value = initialValues[paramName] ?? paramMeta.default;
    
    switch (paramMeta.type) {
      case 'bool':
        return (
          <div className="flex items-center space-x-2">
            <Switch
              id={paramName}
              checked={!!value}
              onCheckedChange={(checked) => onChange(paramName, checked)}
            />
            <Label htmlFor={paramName}>{paramName}</Label>
          </div>
        );
      
      case 'choice':
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramName}</Label>
            <Select
              value={value === null ? 'null' : String(value)}
              onValueChange={(val) => onChange(paramName, val === 'null' ? null : val)}
            >
              <SelectTrigger id={paramName}>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                {paramMeta.choices?.map((choice: any) => (
                  <SelectItem key={String(choice)} value={String(choice)}>
                    {choice === null ? 'None' : String(choice)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );
      
      case 'int':
      case 'float':
        return (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor={paramName}>{paramName}</Label>
              <span className="text-sm text-muted-foreground font-mono">{value}</span>
            </div>
            {paramMeta.min !== undefined && paramMeta.max !== undefined ? (
              <Slider
                value={[value ?? paramMeta.default ?? 0]}
                min={paramMeta.min}
                max={paramMeta.max}
                step={paramMeta.type === 'int' ? 1 : 0.01}
                onValueChange={([val]) => onChange(paramName, val)}
              />
            ) : (
              <Input
                id={paramName}
                type="number"
                value={value ?? ''}
                onChange={(e) => onChange(paramName, paramMeta.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                min={paramMeta.min}
                max={paramMeta.max}
              />
            )}
          </div>
        );
      
      case 'list':
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramName}</Label>
            <Input
              id={paramName}
              value={Array.isArray(value) ? value.join(', ') : ''}
              onChange={(e) => {
                const items = e.target.value.split(',').map((s: string) => {
                  const trimmed = s.trim();
                  const num = Number(trimmed);
                  return isNaN(num) ? trimmed : num;
                });
                onChange(paramName, items);
              }}
              placeholder="Comma-separated values"
            />
          </div>
        );
      
      default:
        return (
          <div className="space-y-1">
            <Label htmlFor={paramName}>{paramName}</Label>
            <Input
              id={paramName}
              value={value ?? ''}
              onChange={(e) => onChange(paramName, e.target.value)}
              placeholder={paramMeta.description}
            />
          </div>
        );
    }
  };

  return (
    <TooltipProvider>
      <Card className="w-full">
        <CardContent className="pt-6 space-y-6">
          {Object.entries(groupedParams).map(([groupName, params]) => (
            <div key={groupName} className="space-y-4">
              <h4 className="font-medium text-sm uppercase tracking-wide text-muted-foreground border-b pb-1">
                {groupName}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(params).map(([paramName, paramMeta]: [string, any]) => (
                  <div key={paramName} className="space-y-1">
                    <div className="flex items-center gap-1">
                      <Label htmlFor={paramName} className="text-sm font-medium">
                        {paramName}
                      </Label>
                      {paramMeta.description && (
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">{paramMeta.description}</p>
                          </TooltipContent>
                        </Tooltip>
                      )}
                    </div>
                    {renderInput(paramName, paramMeta)}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}
