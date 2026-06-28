import { motion } from "framer-motion"

// Big friendly title + a plain-language explanation of what the page is for,
// plus optional right-aligned actions.
export default function PageHeader({ emoji, title, blurb, children }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
      <div>
        <motion.h1
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="font-display text-3xl md:text-4xl font-700 text-ink flex items-center gap-3"
        >
          {emoji && (
            <span className="inline-block animate-bob" aria-hidden>
              {emoji}
            </span>
          )}
          {title}
        </motion.h1>
        {blurb && <p className="text-ink/65 text-lg mt-2 max-w-4xl leading-relaxed">{blurb}</p>}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  )
}
