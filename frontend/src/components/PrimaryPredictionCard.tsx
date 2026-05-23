import { SectionCard } from "./SectionCard";

interface PrimaryPredictionCardProps {
  label: string;
  probability: number;
}

export function PrimaryPredictionCard({
  label,
  probability,
}: PrimaryPredictionCardProps) {
  return (
    <SectionCard
      title="Primary Prediction"
      subtitle="Most likely hypothesis from the selected ML model"
      className="section-card--prediction"
    >
      <div className="prediction-hero">
        <div>
          <p className="prediction-hero__eyebrow">Predicted disease</p>
          <h3 className="prediction-hero__title">{label}</h3>
        </div>
        <div className="prediction-hero__score">
          <span>{Math.round(probability * 100)}%</span>
          <small>confidence</small>
        </div>
      </div>
    </SectionCard>
  );
}
