import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "hsl(36, 33%, 97%)",
        fg: "hsl(204, 30%, 12%)",
        card: "hsl(0, 0%, 100%)",
        accent: "hsl(191, 79%, 35%)",
        warm: "hsl(24, 95%, 52%)",
      },
      boxShadow: {
        soft: "0 18px 45px rgba(26, 58, 70, 0.12)",
      },
      borderRadius: {
        xl: "1rem",
      },
    },
  },
  plugins: [],
};

export default config;
