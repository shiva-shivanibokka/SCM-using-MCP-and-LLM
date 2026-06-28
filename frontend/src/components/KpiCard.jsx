import { motion } from "framer-motion"

const ACCENTS = { teal: "border-teal", amber: "border-amber", coral: "border-coral" }

export default function KpiCard({ title, value, subtitle, accent = "teal" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.03 }}
      className={`bg-white rounded-2xl shadow p-5 border-l-4 ${ACCENTS[accent] || ACCENTS.teal}`}
    >
      <div className="text-xs font-semibold uppercase text-navy/60">{title}</div>
      <div className="text-3xl font-bold text-navy mt-1">{value}</div>
      {subtitle && <div className="text-xs text-navy/50 mt-1">{subtitle}</div>}
    </motion.div>
  )
}
