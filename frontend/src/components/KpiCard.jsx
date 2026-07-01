import { useState } from "react"
import { motion } from "framer-motion"
import { HelpCircle } from "lucide-react"

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

// `help` adds a clickable "?" that pops the explanation ABOVE the title — so it
// never overlaps the number. The full glossary at the page bottom says the same.
export default function KpiCard({ title, value, subtitle, accent = "teal", emoji, help, index = 0, valueClass }) {
  const a = ACCENTS[accent] || ACCENTS.teal
  const [open, setOpen] = useState(false)
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ type: "spring", stiffness: 280, damping: 22, delay: index * 0.06 }}
      whileHover={{ y: -4 }}
      className={`card overflow-visible ring-1 ${a.ring} relative`}
    >
      <div className={`h-2.5 ${a.bar} rounded-t-blob`} />
      <div className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-base font-extrabold uppercase tracking-wide text-ink/55">
            {title}
            {help && (
              <span className="relative inline-flex">
                <button
                  type="button"
                  onClick={() => setOpen((o) => !o)}
                  aria-label={`What does ${title} mean?`}
                  aria-expanded={open}
                  className={`grid place-items-center h-5 w-5 rounded-full transition ${
                    open ? "bg-ink text-white" : "text-ink/35 hover:text-ink hover:bg-ink/5"
                  }`}
                >
                  <HelpCircle size={15} />
                </button>
                {open && (
                  <span
                    role="tooltip"
                    className="absolute bottom-full left-1/2 z-40 mb-2 w-64 -translate-x-1/2 rounded-xl bg-ink px-3 py-2 text-sm font-medium normal-case tracking-normal text-white shadow-pop"
                  >
                    {help}
                    <span className="absolute left-1/2 top-full h-2 w-2 -translate-x-1/2 -translate-y-1 rotate-45 bg-ink" />
                  </span>
                )}
              </span>
            )}
          </div>
          <div className={`grid place-items-center h-12 w-12 rounded-full ${a.tint} text-2xl`}>
            {emoji || a.emoji}
          </div>
        </div>
        <div className={`font-display font-700 text-ink mt-3 leading-none ${valueClass || "text-5xl"}`}>
          {value}
        </div>
        {subtitle && <div className="text-lg text-ink/55 mt-2.5">{subtitle}</div>}
      </div>
    </motion.div>
  )
}
