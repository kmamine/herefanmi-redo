/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        // Figtree for headings, Noto Sans for body (medical / accessible pairing).
        heading: ['Figtree', 'system-ui', 'sans-serif'],
        sans: ['"Noto Sans"', 'system-ui', 'sans-serif'],
      },
      colors: {
        // "Calm cyan + health green" palette from the design system.
        brand: {
          DEFAULT: '#0891B2',
          fg: '#164E63',
          bg: '#ECFEFF',
          border: '#A5F3FC',
          accent: '#059669',
        },
        // Trust-verdict semantic colors (paired with icons, never color alone).
        verdict: {
          trustworthy: '#059669',
          doubtful: '#D97706',
          fake: '#DC2626',
        },
      },
    },
  },
  plugins: [],
};
