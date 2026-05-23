import { SectionCard } from "./SectionCard";

interface ComparisonCardProps {
  mlLabel: string;
  ruleTop1: string;
  ruleTop3: string[];
}

export function ComparisonCard({
  mlLabel,
  ruleTop1,
  ruleTop3,
}: ComparisonCardProps) {
  return (
    <SectionCard
      title="ML vs Rule Engine"
      subtitle="Side-by-side view of the current model and deterministic engine"
    >
      <div className="comparison-grid">
        <div className="comparison-grid__card">
          <p className="comparison-grid__label">ML top 1</p>
          <h3>{mlLabel}</h3>
        </div>
        <div className="comparison-grid__card">
          <p className="comparison-grid__label">Rule-engine top 1</p>
          <h3>{ruleTop1}</h3>
        </div>
      </div>
      <div className="chip-list">
        {ruleTop3.map((label) => (
          <span className="chip" key={label}>
            {label}
          </span>
        ))}
      </div>
    </SectionCard>
  );
}
