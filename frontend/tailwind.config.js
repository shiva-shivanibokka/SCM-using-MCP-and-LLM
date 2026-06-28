/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Petopia playful palette
        cream: "#FFF9F2", // warm biscuit background
        ink: "#2A2140", // deep plum-navy text
        navy: "#2A2140", // alias kept for existing usages
        teal: "#12B5A6", // tail-wag teal
        orange: "#FF7A45", // fetch orange
        amber: "#FFB23E", // sunny (alias amber)
        sunny: "#FFC53D",
        pink: "#FF5DA2", // bubblegum
        sky: "#3DA5F4",
        leaf: "#36C26B",
        grape: "#8B5CF6",
        coral: "#FF6B6B", // risk / danger
      },
      fontFamily: {
        display: ["Fredoka", "system-ui", "sans-serif"],
        sans: ["Nunito", "system-ui", "sans-serif"],
        mono: ["'Space Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        pop: "0 10px 30px -12px rgba(42,33,64,0.25)",
        chunky: "0 6px 0 0 rgba(42,33,64,0.08)",
      },
      borderRadius: { blob: "1.75rem" },
      keyframes: {
        trot: {
          "0%": { transform: "translateX(-10px) scaleX(1)" },
          "49%": { transform: "translateX(120px) scaleX(1)" },
          "50%": { transform: "translateX(120px) scaleX(-1)" },
          "99%": { transform: "translateX(-10px) scaleX(-1)" },
          "100%": { transform: "translateX(-10px) scaleX(1)" },
        },
        bob: {
          "0%,100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-4px)" },
        },
        wiggle: {
          "0%,100%": { transform: "rotate(-8deg)" },
          "50%": { transform: "rotate(8deg)" },
        },
        floatUp: {
          "0%": { transform: "translateY(20px)", opacity: "0" },
          "15%": { opacity: "0.5" },
          "85%": { opacity: "0.5" },
          "100%": { transform: "translateY(-120px)", opacity: "0" },
        },
        popIn: {
          "0%": { transform: "scale(0.9)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },
      animation: {
        trot: "trot 6s ease-in-out infinite",
        bob: "bob 2.4s ease-in-out infinite",
        wiggle: "wiggle 0.6s ease-in-out infinite",
        floatUp: "floatUp 9s linear infinite",
        popIn: "popIn 0.35s ease-out both",
      },
    },
  },
  plugins: [],
}
