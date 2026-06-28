import { motion } from "framer-motion"
import { BookOpen } from "lucide-react"

// Bottom-of-page plain-language key. `items` = [{ term, what }].
export default function Glossary({ items, title = "What these numbers mean" }) {
  if (!items?.length) return null
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="card mt-8 p-6"
    >
      <div className="flex items-center gap-2 mb-4">
        <span className="grid place-items-center h-9 w-9 rounded-full bg-grape/10 text-grape">
          <BookOpen size={18} />
        </span>
        <h2 className="font-display text-xl font-600 text-ink">{title}</h2>
      </div>
      <dl className="grid sm:grid-cols-2 gap-x-10 gap-y-4">
        {items.map((it) => (
          <div key={it.term} className="border-b border-ink/5 pb-3">
            <dt className="font-display font-600 text-ink">{it.term}</dt>
            <dd className="text-sm text-ink/65 mt-0.5 leading-relaxed">{it.what}</dd>
          </div>
        ))}
      </dl>
    </motion.section>
  )
}
