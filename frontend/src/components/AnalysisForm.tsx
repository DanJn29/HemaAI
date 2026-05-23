import { INDICATOR_ORDER, type IndicatorCode } from "../constants/indicators";
import type { AnalysisMode, IndicatorCatalogItem } from "../types/api";

import { SectionCard } from "./SectionCard";

interface AnalysisFormProps {
  sex: "male" | "female";
  age: string;
  mode: AnalysisMode;
  values: Record<IndicatorCode, string>;
  errors: Record<string, string>;
  submitting: boolean;
  indicatorsByCode: Record<string, IndicatorCatalogItem>;
  hasValidationErrors: boolean;
  autoFilledFields: Partial<Record<IndicatorCode, boolean>>;
  autoFilledPatientFields: Partial<Record<"age" | "sex", boolean>>;
  onSexChange: (value: "male" | "female") => void;
  onAgeChange: (value: string) => void;
  onModeChange: (value: AnalysisMode) => void;
  onIndicatorChange: (code: IndicatorCode, value: string) => void;
  onSubmit: () => void;
}

export function AnalysisForm({
  sex,
  age,
  mode,
  values,
  errors,
  submitting,
  indicatorsByCode,
  hasValidationErrors,
  autoFilledFields,
  autoFilledPatientFields,
  onSexChange,
  onAgeChange,
  onModeChange,
  onIndicatorChange,
  onSubmit,
}: AnalysisFormProps) {
  return (
    <SectionCard
      title="CBC Input"
      subtitle="Enter the full panel required by the deployed ML model"
      action={
        <div className="mode-toggle" role="tablist" aria-label="Analysis mode">
          <button
            className={`mode-toggle__button ${mode === "predict" ? "mode-toggle__button--active" : ""}`}
            onClick={() => onModeChange("predict")}
            type="button"
          >
            ML Prediction
          </button>
          <button
            className={`mode-toggle__button ${mode === "compare" ? "mode-toggle__button--active" : ""}`}
            onClick={() => onModeChange("compare")}
            type="button"
          >
            ML + Rule Comparison
          </button>
        </div>
      }
    >
      <div className="form-grid form-grid--patient">
        <label className="field">
          <span className="field__label">
            Sex
            {autoFilledPatientFields.sex ? <span className="field__badge">Auto-filled from image</span> : null}
          </span>
          <select
            className="field__control"
            value={sex}
            onChange={(event) => onSexChange(event.target.value as "male" | "female")}
          >
            <option value="female">Female</option>
            <option value="male">Male</option>
          </select>
        </label>

        <label className="field">
          <span className="field__label">
            Age
            {autoFilledPatientFields.age ? <span className="field__badge">Auto-filled from image</span> : null}
          </span>
          <input
            className={`field__control ${errors.age ? "field__control--error" : ""}`}
            value={age}
            onChange={(event) => onAgeChange(event.target.value)}
            inputMode="numeric"
            placeholder="18-120"
          />
          {errors.age ? <span className="field__error">{errors.age}</span> : null}
        </label>
      </div>

      <div className="field-cluster">
        {INDICATOR_ORDER.map((indicatorCode) => {
          const indicator = indicatorsByCode[indicatorCode];
          const label = indicator?.name || indicatorCode;
          const unit = indicator?.unit || "";
          const description = indicator?.description || "CBC indicator";
          const error = errors[indicatorCode];
          const hasAllowedRange =
            indicator?.min_allowed !== undefined && indicator?.max_allowed !== undefined;
          const hasNormalRange = indicator?.normal_min != null && indicator?.normal_max != null;
          const normalRangeText = `${indicator?.normal_min ?? ""}-${indicator?.normal_max ?? ""}`;
          const allowedRangeText = `${indicator?.min_allowed ?? ""}-${indicator?.max_allowed ?? ""}`;
          const wasAutoFilled = Boolean(autoFilledFields[indicatorCode]);

          return (
            <label className="field" key={indicatorCode}>
              <span className="field__label">
                <strong>{indicatorCode}</strong>
                <span>{label}</span>
                {wasAutoFilled ? <span className="field__badge">Auto-filled from image</span> : null}
              </span>
              <input
                className={`field__control ${error ? "field__control--error" : ""}`}
                value={values[indicatorCode]}
                onChange={(event) => onIndicatorChange(indicatorCode, event.target.value)}
                min={indicator?.min_allowed}
                max={indicator?.max_allowed}
                inputMode="decimal"
                placeholder={unit ? `Value in ${unit}` : "Value"}
                aria-label={`${indicatorCode} value`}
              />
              <span className="field__hint">
                {unit ? `${unit} • ` : ""}{description}
                {hasNormalRange ? (
                  <>
                    <br />
                    Normal: {normalRangeText}
                  </>
                ) : null}
                {hasAllowedRange ? (
                  <>
                    <br />
                    Allowed input: {allowedRangeText}
                  </>
                ) : null}
              </span>
              {error ? <span className="field__error">{error}</span> : null}
            </label>
          );
        })}
      </div>

      <div className="form-actions">
        <button
          className="primary-button"
          onClick={onSubmit}
          disabled={submitting || hasValidationErrors}
          type="button"
        >
          {submitting ? "Analyzing..." : "Run Analysis"}
        </button>
      </div>
    </SectionCard>
  );
}
