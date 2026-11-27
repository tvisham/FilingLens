/** Status is never color alone: a dot plus a text label. */
export function Status({ passed }: { passed: boolean | null | undefined }) {
  const kind = passed === true ? "pass" : passed === false ? "fail" : "skip";
  const label = passed === true ? "Pass" : passed === false ? "Fail" : "Skipped";
  return (
    <span className={`status ${kind}`}>
      <span className="dot" aria-hidden />
      {label}
    </span>
  );
}

export function ErrorBanner({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="banner error" role="alert">
      Could not load data from the FilingLens API: {message}. Is it running? (<code>uv run filinglens api serve</code>)
    </div>
  );
}
