import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  type AdminStats,
  type AdminUser,
  type SourceConfig,
} from "../api/client";
import { ShieldIcon } from "../components/icons";
import { Spinner } from "../components/Spinner";

const BLANK_SOURCE = {
  name: "",
  base_url: "",
  listing_url: "",
  listing_link_selector: "a",
  title_selector: "h1",
  content_selector: "article",
  date_selector: "",
  date_attr: "",
  interval_minutes: 1440,
};

export function AdminPage() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [sources, setSources] = useState<SourceConfig[]>([]);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [queries, setQueries] = useState<Array<Record<string, string>>>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [form, setForm] = useState({ ...BLANK_SOURCE });

  const reload = useCallback(async () => {
    const [s, src, u, q] = await Promise.all([
      api.adminStats().catch(() => null),
      api.adminSources().catch(() => ({ sources: [] })),
      api.adminUsers().catch(() => ({ users: [] })),
      api.adminQueries().catch(() => ({ queries: [] })),
    ]);
    setStats(s);
    setSources(src.sources);
    setUsers(u.users);
    setQueries(q.queries);
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  async function act<T>(fn: () => Promise<T>, label: string) {
    setBusy(true);
    setMsg(null);
    try {
      await fn();
      setMsg(label);
      await reload();
    } catch (e) {
      setMsg(`Error: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (s: SourceConfig) =>
    act(() => api.adminUpdateSource(s.name, { enabled: !s.enabled }), `${s.name} updated`);
  const setInterval = (s: SourceConfig, v: number) =>
    act(() => api.adminUpdateSource(s.name, { interval_minutes: v }), `${s.name} interval set`);
  const remove = (s: SourceConfig) =>
    act(() => api.adminDeleteSource(s.name), `${s.name} removed`);
  const scrapeAll = () => act(() => api.adminScrape(null), "Scrape triggered");
  const addSource = () =>
    act(async () => {
      await api.adminCreateSource({
        ...form,
        date_selector: form.date_selector || null,
        date_attr: form.date_attr || null,
      } as never);
      setForm({ ...BLANK_SOURCE });
    }, "Source added");

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldIcon className="text-brand" width={28} height={28} />
          <h1 className="text-xl font-bold text-brand-fg">Admin</h1>
        </div>
        <Link to="/" className="text-sm text-brand underline underline-offset-2">
          ← Back to checker
        </Link>
      </header>

      {msg && (
        <p className="mb-4 rounded-lg bg-brand-bg px-3 py-2 text-sm text-brand-fg" role="status">
          {busy ? <Spinner /> : null} {msg}
        </p>
      )}

      {/* Knowledge-base stats */}
      <section className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          ["Users", stats?.users],
          ["Queries", stats?.queries],
          ["KB chunks", stats?.chunks],
          ["Sources", stats?.sources],
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl border border-brand-border bg-white p-4">
            <div className="text-2xl font-bold text-brand-fg">{value ?? "—"}</div>
            <div className="text-sm text-brand-fg/70">{label}</div>
          </div>
        ))}
      </section>

      {/* Sources */}
      <section className="mb-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-brand-fg">Sources</h2>
          <button
            type="button"
            onClick={scrapeAll}
            disabled={busy}
            className="rounded-lg bg-brand-accent px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
          >
            Scrape enabled now
          </button>
        </div>
        <div className="overflow-x-auto rounded-xl border border-brand-border bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-brand-bg text-brand-fg/80">
              <tr>
                <th className="px-3 py-2">Source</th>
                <th className="px-3 py-2">Enabled</th>
                <th className="px-3 py-2">Interval (min)</th>
                <th className="px-3 py-2">Last run</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.name} className="border-t border-brand-border">
                  <td className="px-3 py-2 font-medium text-brand-fg">{s.name}</td>
                  <td className="px-3 py-2">
                    <button
                      type="button"
                      onClick={() => toggle(s)}
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        s.enabled ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-700"
                      }`}
                    >
                      {s.enabled ? "Enabled" : "Paused"}
                    </button>
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      defaultValue={s.interval_minutes}
                      min={1}
                      onBlur={(e) => {
                        const v = Number(e.target.value);
                        if (v && v !== s.interval_minutes) setInterval(s, v);
                      }}
                      className="w-24 rounded border border-brand-border px-2 py-1"
                    />
                  </td>
                  <td className="px-3 py-2 text-brand-fg/70">{s.last_run_at ?? "never"}</td>
                  <td className="px-3 py-2 text-brand-fg/70">{s.last_status ?? "—"}</td>
                  <td className="px-3 py-2">
                    <button
                      type="button"
                      onClick={() => remove(s)}
                      className="text-xs text-red-600 underline"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
              {sources.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-4 text-center text-brand-fg/60">
                    No sources configured.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Add source */}
        <details className="mt-4 rounded-xl border border-brand-border bg-white p-4">
          <summary className="cursor-pointer font-medium text-brand-fg">Add a source</summary>
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {(
              [
                ["name", "name (unique)"],
                ["base_url", "base_url"],
                ["listing_url", "listing_url"],
                ["listing_link_selector", "listing_link_selector"],
                ["title_selector", "title_selector"],
                ["content_selector", "content_selector"],
                ["date_selector", "date_selector (optional)"],
                ["date_attr", "date_attr (optional)"],
              ] as const
            ).map(([key, ph]) => (
              <input
                key={key}
                placeholder={ph}
                value={(form as Record<string, string | number>)[key] as string}
                onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                className="rounded border border-brand-border px-2 py-1.5"
              />
            ))}
          </div>
          <button
            type="button"
            onClick={addSource}
            disabled={busy || !form.name || !form.base_url || !form.listing_url}
            className="mt-3 rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-700 disabled:opacity-50"
          >
            Add source
          </button>
        </details>
      </section>

      {/* Users */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold text-brand-fg">Users</h2>
        <div className="overflow-x-auto rounded-xl border border-brand-border bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-brand-bg text-brand-fg/80">
              <tr>
                <th className="px-3 py-2">Email</th>
                <th className="px-3 py-2">Admin</th>
                <th className="px-3 py-2">Points</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.uid} className="border-t border-brand-border">
                  <td className="px-3 py-2 text-brand-fg">{u.email}</td>
                  <td className="px-3 py-2">{u.is_admin ? "✓" : ""}</td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      defaultValue={u.points}
                      onBlur={(e) => {
                        const v = Number(e.target.value);
                        if (v !== u.points) act(() => api.adminSetPoints(u.uid, v), "Points set");
                      }}
                      className="w-24 rounded border border-brand-border px-2 py-1"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Recent queries */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-brand-fg">Recent queries</h2>
        <ul className="space-y-2">
          {queries.map((q) => (
            <li
              key={q.id}
              className="rounded-lg border border-brand-border bg-white px-3 py-2 text-sm"
            >
              <span className="font-medium text-brand-fg">[{q.label}]</span>{" "}
              <span className="text-brand-fg/80">{q.question}</span>
            </li>
          ))}
          {queries.length === 0 && (
            <li className="text-sm text-brand-fg/60">No queries yet.</li>
          )}
        </ul>
      </section>
    </div>
  );
}
