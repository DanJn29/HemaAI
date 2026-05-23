import type { PredictionModelInfo } from "../types/api";

import { SectionCard } from "./SectionCard";

interface ModelInfoPanelProps {
  modelInfo: PredictionModelInfo | null;
}

export function ModelInfoPanel({ modelInfo }: ModelInfoPanelProps) {
  return (
    <SectionCard
      title="Model Snapshot"
      subtitle="Current deployable ML configuration"
    >
      {modelInfo ? (
        <dl className="info-grid">
          <div className="info-grid__item">
            <dt>Model</dt>
            <dd>{modelInfo.model_name}</dd>
          </div>
          <div className="info-grid__item">
            <dt>Dataset</dt>
            <dd>{modelInfo.dataset_variant}</dd>
          </div>
          <div className="info-grid__item">
            <dt>Feature Mode</dt>
            <dd>{modelInfo.feature_mode}</dd>
          </div>
          <div className="info-grid__item">
            <dt>Rule Scores</dt>
            <dd>{modelInfo.include_rule_scores ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>
      ) : (
        <p className="muted-copy">Model metadata is currently unavailable.</p>
      )}
    </SectionCard>
  );
}
