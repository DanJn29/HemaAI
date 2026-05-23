import type { IndicatorCode } from "../constants/indicators";

export interface IndicatorCatalogItem {
  id: number;
  code: string;
  name: string;
  unit: string;
  description: string | null;
  normal_min?: number | null;
  normal_max?: number | null;
  min_allowed?: number;
  max_allowed?: number;
  warning_low?: number | null;
  warning_high?: number | null;
}

export interface DiseaseCatalogItem {
  id: number;
  code: string;
  name: string;
  category: string;
  description: string;
  severity_level: string;
  is_active: boolean;
}

export interface PredictionModelInfo {
  model_name: string;
  dataset_variant: string;
  feature_mode: string;
  include_rule_scores: boolean;
}

export interface ModelMetrics {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  top3_accuracy: number;
}

export interface ModelInfoResponse extends PredictionModelInfo {
  selection_rule: string;
  artifact_format_version: number;
  model_path: string;
  comparison_path: string;
  validation_metrics: ModelMetrics;
  test_metrics: ModelMetrics;
}

export interface PredictionScore {
  label: string;
  probability: number;
}

export interface PredictResponse {
  predicted_label: string;
  top_3_predictions: PredictionScore[];
  model_info: PredictionModelInfo;
}

export interface PredictCompareResponse extends PredictResponse {
  rule_engine: {
    top1_label: string;
    top3_labels: string[];
  };
}

export interface PredictRequest {
  sex: "male" | "female";
  age: number;
  values: Record<IndicatorCode, number>;
}

export interface CbcOcrExtractResponse {
  extracted_values: Record<IndicatorCode, number | null>;
  patient: {
    age: number | null;
    sex: "male" | "female" | null;
  };
  raw_text: string;
  warnings: string[];
}

export type AnalysisMode = "predict" | "compare";
