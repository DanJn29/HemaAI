interface StatusBannerProps {
  tone: "error" | "info";
  message: string;
}

export function StatusBanner({ tone, message }: StatusBannerProps) {
  return <div className={`status-banner status-banner--${tone}`}>{message}</div>;
}
