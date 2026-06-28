import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Suppliers() {
  const { data } = useQuery({
    queryKey: ["suppliers"],
    queryFn: () => apiGet("/api/suppliers/scorecard"),
  })
  const rows = data?.suppliers ?? []
  const cols = rows[0] ? Object.keys(rows[0]) : []
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Suppliers</h1>
      <ChartCard title="Supplier Scorecard (averages across review period)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-navy/50">
                {cols.map((c) => (
                  <th key={c} className="py-1 pr-4">{c.replace(/_/g, " ")}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t">
                  {cols.map((c) => (
                    <td key={c} className="py-1 pr-4">{String(r[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  )
}
