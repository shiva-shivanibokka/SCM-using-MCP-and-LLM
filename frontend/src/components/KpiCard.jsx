import { motion } from "framer-motion"

// Each accent maps to a tile color, a soft tint, and a pet emoji.
const ACCENTS = {
  teal: { bar: "bg-teal", tint: "bg-teal/10", ring: "ring-teal/20", emoji: "🐾" },
  orange: { bar: "bg-orange", tint: "bg-orange/10", ring: "ring-orange/20", emoji: "🦴" },
  amber: { bar: "bg-amber", tint: "bg-amber/10", ring: "ring-amber/20", emoji: "🐥" },
  pink: { bar: "bg-pink", tint: "bg-pink/10", ring: "ring-pink/20", emoji: "🐷" },
  sky: { bar: "bg-sky", tint: "bg-sky/10", ring: "ring-sky/20", emoji: "🐠" },
  leaf: { bar: "bg-leaf", tint: "bg-leaf/10", ring: "ring-leaf/20", emoji: "🐢" },
  grape: { bar: "bg-grape", tint: "bg-grape/10", ring: "ring-grape/20", emoji: "🐙" },
  coral: { bar: "bg-coral", tint: "bg-coral/10", ring: "ring-coral/20", emoji: "🚨" },
}

export default function KpiCard({ title, value, subtitle, accent = "teal", emoji }) {
  const a = ACCENTS[accent] || ACCENTS.teal
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, rotate: -0.4 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`card overflow-hidden ring-1 ${a.ring}`}
    >
      <div className={`h-1.5 ${a.bar}`} />
      <div className="p-5">
        <div className="flex items-center justify-between">
          <div className="text-xs font-extrabold uppercase tracking-wide text-ink/55">
            {title}
          </div>
          <div className={`grid place-items-center h-9 w-9 rounded-full ${a.tint} text-lg`}>
            {emoji || a.emoji}
          </div>
        </div>
        <div className="font-display text-3xl font-700 text-ink mt-2 leading-none">
          {value}
        </div>
        {subtitle && <div className="text-xs text-ink/50 mt-1.5">{subtitle}</div>}
      </div>
    </motion.div>
  )
}
