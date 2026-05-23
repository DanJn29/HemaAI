import { useEffect, useMemo, useState } from "react";

import { AnalysisForm } from "../components/AnalysisForm";
import { ComparisonCard } from "../components/ComparisonCard";
import { EmptyStateCard } from "../components/EmptyStateCard";
import { ModelInfoPanel } from "../components/ModelInfoPanel";
import { OcrUploadCard } from "../components/OcrUploadCard";
import { PrimaryPredictionCard } from "../components/PrimaryPredictionCard";
import { StatusBanner } from "../components/StatusBanner";
import { TopPredictionsCard } from "../components/TopPredictionsCard";
import { INDICATOR_ORDER, type IndicatorCode } from "../constants/indicators";
import {
  extractCbcFromImage,
  fetchDiseases,
  fetchIndicatorMetadata,
  fetchModelInfo,
  submitPrediction,
} from "../services/api";
import type {
  AnalysisMode,
  CbcOcrExtractResponse,
  DiseaseCatalogItem,
  IndicatorCatalogItem,
  ModelInfoResponse,
  PredictCompareResponse,
  PredictResponse,
  PredictionModelInfo,
} from "../types/api";

type FormState = Record<IndicatorCode, string>;
type PatientField = "age" | "sex";

function createEmptyValues(): FormState {
  return INDICATOR_ORDER.reduce(
    (accumulator, indicatorCode) => ({
      ...accumulator,
      [indicatorCode]: "",
    }),
    {} as FormState,
  );
}

export function AnalysisPage() {
  const [sex, setSex] = useState<"male" | "female">("female");
  const [age, setAge] = useState("28");
  const [mode, setMode] = useState<AnalysisMode>("compare");
  const [values, setValues] = useState<FormState>(createEmptyValues);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [autoFilledFields, setAutoFilledFields] = useState<Partial<Record<IndicatorCode, boolean>>>({});
  const [autoFilledPatientFields, setAutoFilledPatientFields] = useState<Partial<Record<PatientField, boolean>>>({});
  const [ocrFile, setOcrFile] = useState<File | null>(null);
  const [ocrPreviewUrl, setOcrPreviewUrl] = useState<string | null>(null);
  const [ocrWarnings, setOcrWarnings] = useState<string[]>([]);
  const [ocrError, setOcrError] = useState<string | null>(null);
  const [ocrUploading, setOcrUploading] = useState(false);
  const [loadingResources, setLoadingResources] = useState(true);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [indicators, setIndicators] = useState<IndicatorCatalogItem[]>([]);
  const [indicatorError, setIndicatorError] = useState<string | null>(null);
  const [diseases, setDiseases] = useState<DiseaseCatalogItem[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [result, setResult] = useState<PredictResponse | PredictCompareResponse | null>(null);

  useEffect(() => {
    let active = true;

    async function loadResources() {
      setLoadingResources(true);
      setResourceError(null);

      const [diseaseResult, modelResult] = await Promise.allSettled([
        fetchDiseases(),
        fetchModelInfo(),
      ]);

      if (!active) {
        return;
      }

      let nextError: string | null = null;

      if (diseaseResult.status === "fulfilled") {
        setDiseases(diseaseResult.value);
      } else {
        nextError =
          diseaseResult.reason instanceof Error
            ? diseaseResult.reason.message
            : "Unable to load disease catalog.";
      }

      if (modelResult.status === "fulfilled") {
        setModelInfo(modelResult.value);
      } else if (nextError === null) {
        nextError =
          modelResult.reason instanceof Error
            ? modelResult.reason.message
            : "Unable to load model metadata.";
      }

      setResourceError(nextError);
      setLoadingResources(false);
    }

    void loadResources();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;

    async function loadIndicatorMetadata() {
      setIndicatorError(null);
      const ageValue = Number(age);
      const hasValidAge = Number.isInteger(ageValue) && ageValue >= 18 && ageValue <= 120;

      try {
        const metadata = hasValidAge
          ? await fetchIndicatorMetadata(sex, ageValue)
          : await fetchIndicatorMetadata();
        if (active) {
          setIndicators(metadata);
        }
      } catch (error) {
        if (active) {
          setIndicatorError(
            error instanceof Error ? error.message : "Unable to load indicator metadata.",
          );
        }
      }
    }

    void loadIndicatorMetadata();

    return () => {
      active = false;
    };
  }, [age, sex]);

  useEffect(() => {
    if (!ocrFile) {
      setOcrPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(ocrFile);
    setOcrPreviewUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [ocrFile]);

  const indicatorsByCode = useMemo(
    () =>
      indicators.reduce<Record<string, IndicatorCatalogItem>>((accumulator, item) => {
        accumulator[item.code] = item;
        return accumulator;
      }, {}),
    [indicators],
  );

  const diseasesByCode = useMemo(
    () =>
      diseases.reduce<Record<string, DiseaseCatalogItem>>((accumulator, item) => {
        accumulator[item.code] = item;
        return accumulator;
      }, {}),
    [diseases],
  );

  const activeModelInfo: PredictionModelInfo | null = result?.model_info || modelInfo;
  const visibleResourceError = resourceError || indicatorError;

  function labelForDisease(code: string) {
    return diseasesByCode[code]?.name || humanizeDiseaseCode(code);
  }

  function updateIndicatorValue(indicatorCode: IndicatorCode, value: string) {
    setValues((current) => ({
      ...current,
      [indicatorCode]: value,
    }));
    setAutoFilledFields((current) => {
      const next = { ...current };
      delete next[indicatorCode];
      return next;
    });
    const error = validateIndicatorValue(indicatorCode, value);
    setFieldErrors((current) => {
      const next = { ...current };
      if (error) {
        next[indicatorCode] = error;
      } else {
        delete next[indicatorCode];
      }
      return next;
    });
  }

  function updateSex(value: "male" | "female") {
    setSex(value);
    setAutoFilledPatientFields((current) => {
      const next = { ...current };
      delete next.sex;
      return next;
    });
  }

  function updateAge(value: string) {
    setAge(value);
    setAutoFilledPatientFields((current) => {
      const next = { ...current };
      delete next.age;
      return next;
    });
    const error = validateAgeValue(value);
    setFieldErrors((current) => {
      const next = { ...current };
      if (error) {
        next.age = error;
      } else {
        delete next.age;
      }
      return next;
    });
  }

  function validateForm() {
    const nextErrors: Record<string, string> = {};

    const ageError = validateAgeValue(age);
    if (ageError) {
      nextErrors.age = ageError;
    }

    for (const indicatorCode of INDICATOR_ORDER) {
      const raw = values[indicatorCode].trim();
      if (!raw) {
        nextErrors[indicatorCode] = "This value is required.";
        continue;
      }
      const indicatorError = validateIndicatorValue(indicatorCode, raw);
      if (indicatorError) {
        nextErrors[indicatorCode] = indicatorError;
      }
    }

    return nextErrors;
  }

  function validateAgeValue(rawValue: string) {
    const ageValue = Number(rawValue);
    if (!Number.isFinite(ageValue) || !Number.isInteger(ageValue)) {
      return "Enter an integer age.";
    }
    if (ageValue < 18 || ageValue > 120) {
      return "Age must be between 18 and 120.";
    }
    return "";
  }

  function validateIndicatorValue(indicatorCode: IndicatorCode, rawValue: string) {
    const raw = rawValue.trim();
    if (!raw) {
      return "";
    }

    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      return "Enter a valid number.";
    }

    const indicator = indicatorsByCode[indicatorCode];
    if (
      indicator?.min_allowed !== undefined &&
      indicator?.max_allowed !== undefined &&
      (parsed < indicator.min_allowed || parsed > indicator.max_allowed)
    ) {
      const unitSuffix = indicator.unit ? ` ${indicator.unit}` : "";
      return `${indicatorCode} must be between ${formatNumber(indicator.min_allowed)} and ${formatNumber(
        indicator.max_allowed,
      )}${unitSuffix}.`;
    }

    return "";
  }

  async function handleSubmit() {
    const nextErrors = validateForm();
    setFieldErrors(nextErrors);
    setSubmitError(null);

    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    setSubmitting(true);

    try {
      const payload = {
        sex,
        age: Number(age),
        values: INDICATOR_ORDER.reduce<Record<IndicatorCode, number>>((accumulator, indicatorCode) => {
          accumulator[indicatorCode] = Number(values[indicatorCode]);
          return accumulator;
        }, {} as Record<IndicatorCode, number>),
      };
      const response = await submitPrediction(mode, payload);
      setResult(response);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Unable to complete the analysis.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleOcrFileChange(file: File | null) {
    setOcrFile(file);
    setOcrError(null);
    setOcrWarnings([]);
  }

  async function handleOcrUpload() {
    if (!ocrFile) {
      return;
    }

    setOcrUploading(true);
    setOcrError(null);
    setOcrWarnings([]);

    try {
      const response = await extractCbcFromImage(ocrFile);
      applyOcrResult(response);
    } catch (error) {
      setOcrError(error instanceof Error ? error.message : "Unable to extract CBC values from image.");
    } finally {
      setOcrUploading(false);
    }
  }

  function applyOcrResult(response: CbcOcrExtractResponse) {
    const nextAutoFilled: Partial<Record<IndicatorCode, boolean>> = {};
    const nextAutoFilledPatientFields: Partial<Record<PatientField, boolean>> = {};
    const nextValues = { ...values };
    const nextErrors = { ...fieldErrors };

    if (response.patient.sex === "male" || response.patient.sex === "female") {
      setSex(response.patient.sex);
      nextAutoFilledPatientFields.sex = true;
    }

    if (response.patient.age !== null && response.patient.age !== undefined) {
      const ageText = String(response.patient.age);
      setAge(ageText);
      nextAutoFilledPatientFields.age = true;
      const ageError = validateAgeValue(ageText);
      if (ageError) {
        nextErrors.age = ageError;
      } else {
        delete nextErrors.age;
      }
    }

    for (const indicatorCode of INDICATOR_ORDER) {
      const extractedValue = response.extracted_values[indicatorCode];
      if (extractedValue === null || extractedValue === undefined) {
        continue;
      }

      const valueText = formatNumber(extractedValue);
      nextValues[indicatorCode] = valueText;
      nextAutoFilled[indicatorCode] = true;

      const error = validateIndicatorValue(indicatorCode, valueText);
      if (error) {
        nextErrors[indicatorCode] = error;
      } else {
        delete nextErrors[indicatorCode];
      }
    }

    setValues(nextValues);
    setAutoFilledFields(nextAutoFilled);
    setAutoFilledPatientFields(nextAutoFilledPatientFields);
    setFieldErrors(nextErrors);
    setOcrWarnings(response.warnings);
    setSubmitError(null);
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-panel__content">
          <span className="hero-panel__badge">Clinical AI decision support</span>
          <h1>HemaAI</h1>
          <p>
            Evaluate a full CBC panel with the current deployable ML model and
            compare it with the existing rule-based engine without leaving the dashboard.
          </p>
        </div>
      </section>

      <section className="workspace">
        <div className="workspace__column workspace__column--form">
          {visibleResourceError ? <StatusBanner tone="error" message={visibleResourceError} /> : null}
          {submitError ? <StatusBanner tone="error" message={submitError} /> : null}

          <OcrUploadCard
            selectedFile={ocrFile}
            previewUrl={ocrPreviewUrl}
            uploading={ocrUploading}
            warnings={ocrWarnings}
            error={ocrError}
            onFileChange={handleOcrFileChange}
            onUpload={handleOcrUpload}
          />

          <AnalysisForm
            sex={sex}
            age={age}
            mode={mode}
            values={values}
            errors={fieldErrors}
            submitting={submitting}
            indicatorsByCode={indicatorsByCode}
            hasValidationErrors={Object.keys(fieldErrors).length > 0}
            autoFilledFields={autoFilledFields}
            autoFilledPatientFields={autoFilledPatientFields}
            onSexChange={updateSex}
            onAgeChange={updateAge}
            onModeChange={setMode}
            onIndicatorChange={updateIndicatorValue}
            onSubmit={handleSubmit}
          />
        </div>

        <div className="workspace__column workspace__column--results">
          <ModelInfoPanel modelInfo={activeModelInfo} />

          {loadingResources ? (
            <EmptyStateCard />
          ) : result ? (
            <>
              <PrimaryPredictionCard
                label={labelForDisease(result.predicted_label)}
                probability={result.top_3_predictions[0]?.probability ?? 0}
              />
              <TopPredictionsCard
                predictions={result.top_3_predictions}
                labelFor={labelForDisease}
              />
              {"rule_engine" in result ? (
                <ComparisonCard
                  mlLabel={labelForDisease(result.predicted_label)}
                  ruleTop1={labelForDisease(result.rule_engine.top1_label)}
                  ruleTop3={result.rule_engine.top3_labels.map(labelForDisease)}
                />
              ) : null}
            </>
          ) : (
            <EmptyStateCard />
          )}
        </div>
      </section>
    </main>
  );
}

function humanizeDiseaseCode(code: string) {
  return code
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? String(value) : String(value);
}
