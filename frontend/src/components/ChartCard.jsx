export default function ChartCard({ title, children }) {
  return (
    <div className="bg-white rounded-2xl shadow p-5">
      <div className="text-sm font-semibold text-navy mb-3">{title}</div>
      {children}
    </div>
  )
}
