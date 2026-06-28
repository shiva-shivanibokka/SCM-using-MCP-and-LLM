import { motion } from "framer-motion"
import { Info } from "lucide-react"

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

// `help` shows an info dot with a hover tooltip explaining what the metric means.
export default function KpiCard({ title, value, subtitle, accent = "teal", emoji, help, index = 0 }) {
  const a = ACCENTS[accent] || ACCENTS.teal
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ type: "spring", stiffness: 280, damping: 22, delay: index * 0.06 }}
      whileHover={{ y: -5, rotate: -0.5 }}
      className={`group card overflow-visible ring-1 ${a.ring} relative`}
    >
      <div className={`h-2 ${a.bar} rounded-t-blob`} />
      <div className="p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-sm font-extrabold uppercase tracking-wide text-ink/55">
            {title}
            {help && (
              <span className="relative">
                <Info size={14} className="text-ink/30 group-hover:text-ink/60 transition" />
                <span className="pointer-events-none absolute left-1/2 top-6 z-20 w-56 -translate-x-1/2 rounded-xl bg-ink px-3 py-2 text-xs font-medium normal-case tracking-normal text-white opacity-0 shadow-pop transition group-hover:opacity-100">
                  {help}
                </span>
              </span>
            )}
          </div>
          <div className={`grid place-items-center h-10 w-10 rounded-full ${a.tint} text-xl`}>
            {emoji || a.emoji}
          </div>
        </div>
        <div className="font-display text-4xl font-700 text-ink mt-2 leading-none">
          {value}
        </div>
        {subtitle && <div className="text-sm text-ink/55 mt-2">{subtitle}</div>}
      </div>
    </motion.div>
  )
}
