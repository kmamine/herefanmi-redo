import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import {
  api,
  clearToken,
  getAdminFlag,
  getToken,
  setAdminFlag,
  setToken,
} from "../api/client";

interface AuthState {
  uid: string | null;
  points: number;
  isAdmin: boolean;
  isAuthed: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
  setPoints: (p: number) => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [uid, setUid] = useState<string | null>(null);
  const [points, setPoints] = useState<number>(0);
  const [isAdmin, setIsAdmin] = useState<boolean>(getAdminFlag());
  const [authed, setAuthed] = useState<boolean>(Boolean(getToken()));

  const value = useMemo<AuthState>(
    () => ({
      uid,
      points,
      isAdmin,
      isAuthed: authed,
      async login(email, password) {
        const r = await api.login(email, password);
        setToken(r.access_token);
        setUid(r.uid);
        setPoints(r.points);
        setIsAdmin(r.is_admin);
        setAdminFlag(r.is_admin);
        setAuthed(true);
      },
      async signup(email, password) {
        const r = await api.signup(email, password);
        setToken(r.access_token);
        setUid(r.uid);
        setPoints(r.points);
        setIsAdmin(r.is_admin);
        setAdminFlag(r.is_admin);
        setAuthed(true);
      },
      logout() {
        clearToken();
        setUid(null);
        setPoints(0);
        setIsAdmin(false);
        setAuthed(false);
      },
      setPoints,
    }),
    [uid, points, isAdmin, authed],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
