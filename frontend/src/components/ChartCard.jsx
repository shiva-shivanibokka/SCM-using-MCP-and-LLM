// A titled white panel. `hint` explains what the chart/table means (plain
// language), `action` renders controls on the right (selectors, buttons).
export default function ChartCard({ title, hint, action, children, className = "" }) {
  return (
    <div className={`card p-5 ${className}`}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="font-display text-lg font-600 text-ink">{title}</div>
          {hint && <div className="text-xs text-ink/50 mt-0.5 max-w-prose">{hint}</div>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {children}
    </div>
  )
}
