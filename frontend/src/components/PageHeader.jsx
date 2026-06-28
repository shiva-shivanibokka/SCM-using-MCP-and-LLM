import { motion } from "framer-motion"

// Big friendly title + a plain-language explanation of what the page is for,
// plus optional right-aligned actions. The blurb spans the full content width.
export default function PageHeader({ emoji, title, blurb, children }) {
  return (
    <div className="mb-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <motion.h1
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          className="font-display text-4xl md:text-5xl font-700 text-ink flex items-center gap-3"
        >
          {emoji && (
            <span className="inline-block animate-bob text-5xl" aria-hidden>
              {emoji}
            </span>
          )}
          {title}
        </motion.h1>
        {children && <div className="flex items-center gap-2">{children}</div>}
      </div>
      {blurb && (
        <p className="text-ink/65 text-xl mt-3 leading-relaxed">{blurb}</p>
      )}
    </div>
  )
}
