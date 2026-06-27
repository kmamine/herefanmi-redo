import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type MedicalTalkResult } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ResultCard } from "../components/ResultCard";
import { ShieldIcon } from "../components/icons";
import { Spinner } from "../components/Spinner";
import { StarRating } from "../components/StarRating";

const BACKENDS = [
  { label: "HeReFaNMi LLM", enabled: true },
  { label: "GPT-4", enabled: false },
  { label: "GPT-3.5", enabled: false },
  { label: "Mistral", enabled: false },
];

export function HomePage() {
  const { points, setPoints, logout, isAdmin } = useAuth();
  const [phase, setPhase] = useState<1 | 2>(1);
  const [text, setText] = useState("");
  const [opinion, setOpinion] = useState(0);
  const [backend, setBackend] = useState("HeReFaNMi LLM");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MedicalTalkResult | null>(null);
  const [rating, setRating] = useState(0);

  // Token persists across reloads but in-memory points don't — refresh on mount.
  useEffect(() => {
    api
      .pointCheck()
      .then((p) => setPoints(p.points))
      .catch(() => {});
  }, [setPoints]);

  function reset() {
    setPhase(1);
    setText("");
    setOpinion(0);
    setResult(null);
    setRating(0);
    setError(null);
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const r = await api.medicalTalk(text, String(opinion), backend);
      setResult(r);
      // Refresh points after a successful (charged) query.
      try {
        const p = await api.pointCheck();
        setPoints(p.points);
      } catch {
        /* ignore */
      }
    } catch (err) {
      const e = err as Error & { status?: number };
      setError(e.status === 403 ? "You are out of query points." : e.message);
    } finally {
      setBusy(false);
    }
  }

  async function submitRating(value: number) {
    if (!result) return;
    setRating(value);
    try {
      await api.save(result.key, String(value));
    } catch {
      /* non-blocking */
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-6">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldIcon className="text-brand" width={28} height={28} />
          <h1 className="text-xl font-bold text-brand-fg">HeReFaNMi</h1>
        </div>
        <div className="flex items-center gap-4">
          {isAdmin && (
            <Link
              to="/admin"
              className="text-sm font-medium text-brand underline underline-offset-2"
            >
              Admin
            </Link>
          )}
          <span className="rounded-full bg-brand-bg px-3 py-1 text-sm font-medium text-brand-fg">
            {points} points
          </span>
          <button
            type="button"
            onClick={logout}
            className="text-sm text-brand underline underline-offset-2"
          >
            Sign out
          </button>
        </div>
      </header>

      <p className="mb-6 text-brand-fg/80">
        Paste a health-related claim or question. HeReFaNMi checks it against
        credible medical sources and classifies it as{" "}
        <span className="font-semibold text-emerald-700">Trustworthy</span>,{" "}
        <span className="font-semibold text-amber-700">Doubtful</span>, or{" "}
        <span className="font-semibold text-red-700">Fake</span>.
      </p>

      {phase === 1 && (
        <section className="rounded-2xl border border-brand-border bg-white p-5">
          <label htmlFor="claim" className="block font-medium text-brand-fg">
            Health claim or question
          </label>
          <textarea
            id="claim"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. Does vitamin D help bone health?"
            className="mt-2 w-full resize-y rounded-lg border border-brand-border px-3 py-2.5 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/30"
          />

          <div className="mt-4">
            <span className="block font-medium text-brand-fg">
              How much do you trust this claim?
            </span>
            <p className="text-sm text-brand-fg/60">Optional — your own opinion (1–5)</p>
            <div className="mt-1">
              <StarRating value={opinion} onChange={setOpinion} label="Your confidence" />
            </div>
          </div>

          <button
            type="button"
            disabled={!text.trim()}
            onClick={() => setPhase(2)}
            className="mt-5 rounded-lg bg-brand px-5 py-2.5 font-semibold text-white transition-colors hover:bg-cyan-700 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/40 disabled:opacity-50"
          >
            Continue
          </button>
        </section>
      )}

      {phase === 2 && (
        <section className="space-y-5">
          <div className="rounded-2xl border border-brand-border bg-white p-5">
            <p className="text-sm text-brand-fg/70">Your claim</p>
            <p className="mt-1 text-brand-fg">{text}</p>

            <label htmlFor="backend" className="mt-4 block font-medium text-brand-fg">
              Analysis model
            </label>
            <select
              id="backend"
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              className="mt-1 w-full rounded-lg border border-brand-border px-3 py-2.5 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/30"
            >
              {BACKENDS.map((b) => (
                <option key={b.label} value={b.label} disabled={!b.enabled}>
                  {b.label}
                  {b.enabled ? "" : " (coming soon)"}
                </option>
              ))}
            </select>

            {error && (
              <p role="alert" className="mt-3 text-sm text-red-600">
                {error}
              </p>
            )}

            <div className="mt-5 flex gap-3">
              <button
                type="button"
                onClick={submit}
                disabled={busy}
                className="flex items-center justify-center gap-2 rounded-lg bg-brand-accent px-5 py-2.5 font-semibold text-white transition-colors hover:bg-emerald-700 focus:outline-none focus-visible:ring-4 focus-visible:ring-emerald-500/40 disabled:opacity-60"
              >
                {busy && <Spinner />}
                {busy ? "Checking…" : "Check claim"}
              </button>
              <button
                type="button"
                onClick={reset}
                className="rounded-lg border border-brand-border px-5 py-2.5 font-medium text-brand-fg hover:bg-brand-bg"
              >
                Start over
              </button>
            </div>
          </div>

          {result && (
            <>
              <ResultCard result={result} />
              {result.label && (
                <div className="flex flex-wrap items-center gap-4 rounded-xl border border-brand-border bg-white p-4">
                  <span className="text-sm font-medium text-brand-fg">
                    Rate this answer:
                  </span>
                  <StarRating value={rating} onChange={submitRating} label="Rate the answer" />
                  <button
                    type="button"
                    onClick={() => navigator.clipboard?.writeText(result.data)}
                    className="ml-auto text-sm text-brand underline underline-offset-2"
                  >
                    Copy
                  </button>
                  <button
                    type="button"
                    onClick={reset}
                    className="text-sm text-brand underline underline-offset-2"
                  >
                    New check
                  </button>
                </div>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}
