export const INDICATOR_ORDER = [
  "WBC",
  "RBC",
  "HGB",
  "HCT",
  "MCV",
  "MCH",
  "MCHC",
  "PLT",
  "RDW",
  "NEU",
  "LYM",
  "MONO",
  "EOS",
  "BASO",
] as const;

export type IndicatorCode = (typeof INDICATOR_ORDER)[number];
