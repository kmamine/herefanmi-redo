import type { MedicalTalkResult } from "../api/client";
import { ExternalLink } from "./icons";
import { VERDICTS } from "./verdict";

interface ResultCardProps {
  result: MedicalTalkResult;
}

// Renders the classification verdict. Non-medical responses have no label and
// are shown as a neutral notice.
export function ResultCard({ result }: ResultCardProps) {
  if (!result.label) {
    return (
      <div
        role="status"
        className="rounded-xl border border-brand-border bg-white p-5 text-brand-fg"
      >
        {result.data}
      </div>
    );
  }

  const v = VERDICTS[result.label];
  const { Icon } = v;
  const sources = result.source ?? [];

  return (
    <article
      role="status"
      aria-live="polite"
      className={`rounded-xl border p-5 ${v.card}`}
    >
      <header className="flex items-center gap-3">
        <Icon className={v.text} width={28} height={28} />
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${v.badge}`}
        >
          {v.label}
        </span>
        {result.news === "True" && (
          <span className="text-sm text-brand-fg/70">News / claim</span>
        )}
      </header>

      <p className="mt-4 leading-relaxed text-brand-fg">{result.data}</p>

      {sources.length > 0 && (
        <section className="mt-5">
          <h3 className="text-sm font-semibold text-brand-fg/80">Sources</h3>
          <ul className="mt-2 space-y-1">
            {sources.map((url) => (
              <li key={url}>
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-brand underline decoration-brand/40 underline-offset-2 hover:decoration-brand"
                >
                  <ExternalLink />
                  <span className="break-all">{url}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
