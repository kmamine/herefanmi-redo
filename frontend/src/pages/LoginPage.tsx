import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { ShieldIcon } from "../components/icons";
import { Spinner } from "../components/Spinner";

export function LoginPage() {
  const { login, signup } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "signup") await signup(email, password);
      else await login(email, password);
      navigate("/");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-md flex-col justify-center px-4">
      <div className="mb-6 flex items-center gap-3">
        <ShieldIcon className="text-brand" width={32} height={32} />
        <div>
          <h1 className="text-2xl font-bold text-brand-fg">HeReFaNMi</h1>
          <p className="text-sm text-brand-fg/70">Health-Related Fake News checker</p>
        </div>
      </div>

      <form
        onSubmit={onSubmit}
        className="rounded-2xl border border-brand-border bg-white p-6 shadow-sm"
      >
        <h2 className="mb-4 text-lg font-semibold text-brand-fg">
          {mode === "login" ? "Sign in" : "Create an account"}
        </h2>

        <label className="block text-sm font-medium text-brand-fg" htmlFor="email">
          Email
        </label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mb-4 mt-1 w-full rounded-lg border border-brand-border px-3 py-2.5 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/30"
        />

        <label className="block text-sm font-medium text-brand-fg" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          required
          minLength={6}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mb-4 mt-1 w-full rounded-lg border border-brand-border px-3 py-2.5 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/30"
        />

        {error && (
          <p role="alert" className="mb-3 text-sm text-red-600">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand px-4 py-2.5 font-semibold text-white transition-colors hover:bg-cyan-700 focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/40 disabled:opacity-60"
        >
          {busy && <Spinner />}
          {mode === "login" ? "Sign in" : "Sign up"}
        </button>
      </form>

      <button
        type="button"
        onClick={() => {
          setMode(mode === "login" ? "signup" : "login");
          setError(null);
        }}
        className="mt-4 text-sm text-brand underline underline-offset-2"
      >
        {mode === "login"
          ? "New here? Create an account"
          : "Already have an account? Sign in"}
      </button>
    </main>
  );
}
