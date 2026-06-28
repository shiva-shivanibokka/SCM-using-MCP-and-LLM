// Shared formatting helpers.

export const inr = (n) =>
  "₹" + Number(n ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })

// Compact INR for big numbers: ₹49.4Cr, ₹2.3L
export const inrCompact = (n) => {
  const v = Number(n ?? 0)
  if (v >= 1e7) return "₹" + (v / 1e7).toFixed(1) + "Cr"
  if (v >= 1e5) return "₹" + (v / 1e5).toFixed(1) + "L"
  if (v >= 1e3) return "₹" + (v / 1e3).toFixed(1) + "K"
  return "₹" + v.toFixed(0)
}

export const num = (n) =>
  Number(n ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 1 })

export const pct = (n) => `${Number(n ?? 0).toFixed(1)}%`

export const titleize = (s) =>
  String(s ?? "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
