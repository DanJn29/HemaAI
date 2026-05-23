import { SectionCard } from "./SectionCard";

export function EmptyStateCard() {
  return (
    <SectionCard
      title="Ready for Analysis"
      subtitle="Enter a full CBC panel to generate a prediction"
    >
      <p className="muted-copy">
        The dashboard uses the live HemaAI backend and the currently selected
        deployable model. Submit a complete CBC profile to view probabilities,
        top-three hypotheses, and optional rule-engine comparison.
      </p>
    </SectionCard>
  );
}
