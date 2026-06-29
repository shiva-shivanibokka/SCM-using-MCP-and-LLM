import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

// Tailwind resets default element styles, so we map each markdown element to
// classed components. Block code is styled on <pre> (v9 has no `inline` prop).
const COMPONENTS = {
  p: (p) => <p className="mb-2 last:mb-0 leading-relaxed" {...p} />,
  ul: (p) => <ul className="list-disc pl-5 mb-2 space-y-1" {...p} />,
  ol: (p) => <ol className="list-decimal pl-5 mb-2 space-y-1" {...p} />,
  li: (p) => <li className="leading-relaxed" {...p} />,
  strong: (p) => <strong className="font-bold text-ink" {...p} />,
  em: (p) => <em className="italic" {...p} />,
  a: (p) => <a className="text-teal underline underline-offset-2" target="_blank" rel="noreferrer" {...p} />,
  h1: (p) => <h1 className="font-display font-700 text-lg mt-1 mb-2" {...p} />,
  h2: (p) => <h2 className="font-display font-700 text-base mt-1 mb-2" {...p} />,
  h3: (p) => <h3 className="font-bold text-sm mt-1 mb-1" {...p} />,
  hr: () => <hr className="my-3 border-ink/10" />,
  blockquote: (p) => <blockquote className="border-l-2 border-ink/20 pl-3 italic text-ink/70 my-2" {...p} />,
  code: (p) => <code className="rounded bg-ink/10 px-1 py-0.5 font-mono text-[0.85em]" {...p} />,
  pre: (p) => (
    <pre
      className="my-2 overflow-x-auto rounded-lg bg-ink/90 p-3 font-mono text-xs text-white [&>code]:bg-transparent [&>code]:p-0 [&>code]:text-white"
      {...p}
    />
  ),
  table: (p) => <table className="my-2 w-full border-collapse text-xs" {...p} />,
  th: (p) => <th className="border border-ink/15 bg-ink/5 px-2 py-1 text-left font-bold" {...p} />,
  td: (p) => <td className="border border-ink/15 px-2 py-1" {...p} />,
}

export default function Markdown({ children, className = "" }) {
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children || ""}
      </ReactMarkdown>
    </div>
  )
}
