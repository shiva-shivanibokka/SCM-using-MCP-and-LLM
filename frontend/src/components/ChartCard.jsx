import { motion } from "framer-motion"

// A titled white panel. `hint` explains what the chart/table means (plain
// language), `action` renders controls on the right (selectors, buttons).
export default function ChartCard({ title, hint, action, children, className = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.4 }}
      className={`card p-6 ${className}`}
    >
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <div className="font-display text-xl font-600 text-ink">{title}</div>
          {hint && <div className="text-sm text-ink/55 mt-1 max-w-prose">{hint}</div>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {children}
    </motion.div>
  )
}
