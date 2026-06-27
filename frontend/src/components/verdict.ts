// Trust-verdict visual config: color is always paired with an icon + label
// (WCAG: never convey meaning by color alone).
import type { ComponentType, SVGProps } from "react";
import type { Verdict } from "../api/client";
import { AlertTriangle, CheckCircle, XCircle } from "./icons";

export interface VerdictStyle {
  label: string;
  Icon: ComponentType<SVGProps<SVGSVGElement>>;
  card: string; // border + background tint
  badge: string; // badge background + text
  text: string; // icon/title color
}

export const VERDICTS: Record<Verdict, VerdictStyle> = {
  Trustworthy: {
    label: "Trustworthy",
    Icon: CheckCircle,
    card: "border-emerald-300 bg-emerald-50",
    badge: "bg-emerald-600 text-white",
    text: "text-emerald-700",
  },
  Doubtful: {
    label: "Doubtful",
    Icon: AlertTriangle,
    card: "border-amber-300 bg-amber-50",
    badge: "bg-amber-500 text-white",
    text: "text-amber-700",
  },
  Fake: {
    label: "Fake",
    Icon: XCircle,
    card: "border-red-300 bg-red-50",
    badge: "bg-red-600 text-white",
    text: "text-red-700",
  },
};
