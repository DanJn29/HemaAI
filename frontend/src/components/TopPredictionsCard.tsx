import type { PredictionScore } from "../types/api";

import { SectionCard } from "./SectionCard";

interface TopPredictionsCardProps {
  predictions: PredictionScore[];
  labelFor: (code: string) => string;
}

export function TopPredictionsCard({
  predictions,
  labelFor,
}: TopPredictionsCardProps) {
  return (
    <SectionCard title="Top 3 Hypotheses" subtitle="Ranked by model probability">
      <div className="prediction-list">
        {predictions.map((prediction, index) => (
          <div className="prediction-list__item" key={prediction.label}>
            <div className="prediction-list__header">
              <span className="prediction-list__rank">#{index + 1}</span>
              <span className="prediction-list__name">
                {labelFor(prediction.label)}
              </span>
              <span className="prediction-list__value">
                {(prediction.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="progress-bar">
              <span
                className="progress-bar__fill"
                style={{ width: `${Math.max(prediction.probability * 100, 4)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}
