// frontend_next/src/types/ml.ts

export enum UIInputType {
  TEXT = "text",
  NUMBER = "number",
  INTEGER = "integer",
  BOOLEAN = "boolean",
  SELECT = "select",
  MULTI_SELECT = "multi_select",
  SLIDER = "slider",
  TEXTAREA = "textarea",
  FILE = "file",
  COLOR = "color",
  DATE = "date",
  DATETIME = "datetime"
}

export interface UIParamMetadata {
  name: string;
  label?: string;
  input_type: UIInputType;
  default?: any;
  required?: boolean;
  description?: string;
  placeholder?: string;
  min_value?: number;
  max_value?: number;
  step?: number;
  options?: Array<{ value: any; label: string; description?: string }>;
  pattern?: string;
  min_length?: number;
  max_length?: number;
  hidden?: boolean;
  disabled?: boolean;
  depends_on?: Record<string, any>;
  advanced?: boolean;
  category?: string;
  order?: number;
}

export type EstimatorUIMetadata = Record<string, UIParamMetadata>;
