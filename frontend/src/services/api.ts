import type {
  AnalysisMode,
  CbcOcrExtractResponse,
  DiseaseCatalogItem,
  IndicatorCatalogItem,
  ModelInfoResponse,
  PredictCompareResponse,
  PredictRequest,
  PredictResponse,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(formatApiError(payload));
  }

  return response.json() as Promise<T>;
}

function formatApiError(payload: unknown) {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return "The request failed. Please check the form values and try again.";
  }

  const detail = (payload as { detail: unknown }).detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object" && "errors" in detail) {
    const validationDetail = detail as { message?: unknown; errors?: unknown };
    if (Array.isArray(validationDetail.errors)) {
      const messages = validationDetail.errors
        .map((item) =>
          item && typeof item === "object" && "message" in item
            ? String((item as { message: unknown }).message)
            : "",
        )
        .filter(Boolean);
      if (messages.length > 0) {
        return messages.join(" ");
      }
    }
    if (typeof validationDetail.message === "string") {
      return validationDetail.message;
    }
  }

  return "The request failed. Please check the form values and try again.";
}

export function fetchIndicators() {
  return apiRequest<IndicatorCatalogItem[]>("/indicators");
}

export function fetchIndicatorMetadata(sex?: "male" | "female", age?: number) {
  const searchParams = new URLSearchParams();
  if (sex && age !== undefined) {
    searchParams.set("sex", sex);
    searchParams.set("age", String(age));
  }
  const queryString = searchParams.toString();
  return apiRequest<IndicatorCatalogItem[]>(
    queryString ? `/indicators/metadata?${queryString}` : "/indicators/metadata",
  );
}

export function fetchDiseases() {
  return apiRequest<DiseaseCatalogItem[]>("/diseases");
}

export function fetchModelInfo() {
  return apiRequest<ModelInfoResponse>("/ml/model-info");
}

export function submitPrediction(
  mode: AnalysisMode,
  payload: PredictRequest,
) {
  const path = mode === "compare" ? "/ml/predict-and-compare" : "/ml/predict";
  return apiRequest<PredictResponse | PredictCompareResponse>(path, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function extractCbcFromImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/ocr/cbc-extract`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(formatApiError(payload));
  }

  return response.json() as Promise<CbcOcrExtractResponse>;
}
