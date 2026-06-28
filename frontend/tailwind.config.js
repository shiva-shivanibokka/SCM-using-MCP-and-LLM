/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        teal: "#0D9488",
        amber: "#F59E0B",
        coral: "#F87171",
        navy: "#1E293B",
        cream: "#FAFAF7",
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
}
