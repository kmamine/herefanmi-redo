interface StarRatingProps {
  value: number;
  onChange: (value: number) => void;
  label: string;
  disabled?: boolean;
}

// Accessible 1-5 star control: radio-group semantics, keyboard focusable,
// ≥44px touch targets.
export function StarRating({ value, onChange, label, disabled }: StarRatingProps) {
  return (
    <div role="radiogroup" aria-label={label} className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => {
        const active = n <= value;
        return (
          <button
            key={n}
            type="button"
            role="radio"
            aria-checked={value === n}
            aria-label={`${n} star${n > 1 ? "s" : ""}`}
            disabled={disabled}
            onClick={() => onChange(n)}
            className="flex h-11 w-11 items-center justify-center rounded-md transition-colors hover:bg-brand-bg focus:outline-none focus-visible:ring-4 focus-visible:ring-brand/40 disabled:cursor-not-allowed"
          >
            <svg
              width="26"
              height="26"
              viewBox="0 0 24 24"
              fill={active ? "#0891B2" : "none"}
              stroke={active ? "#0891B2" : "#94A3B8"}
              strokeWidth="1.75"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8L4.5 9.7l5.9-.9L12 3.5Z" />
            </svg>
          </button>
        );
      })}
    </div>
  );
}
