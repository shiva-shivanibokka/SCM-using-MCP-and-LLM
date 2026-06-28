import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"

function Table({ rows }) {
  if (!rows?.length) return <div className="text-navy/40 text-sm">Nothing here 🎉</div>
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-navy/50">
          <th className="py-1">SKU</th>
          <th>Inventory</th>
          <th>Days of cover</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.sku_id} className="border-t">
            <td className="py-1">{r.sku_id}</td>
            <td>{r.inventory}</td>
            <td>{r.days_of_cover}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function Inventory() {
  const { data, isLoading } = useQuery({
    queryKey: ["inv-health"],
    queryFn: () => apiGet("/api/inventory/health"),
  })
  if (isLoading || !data) return <div className="text-navy/50">Loading…</div>
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Inventory Health</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard title="Reorder" value={data.reorder_list.length} accent="amber" />
        <KpiCard title="Stockout Risk" value={data.stockout_risk.length} accent="coral" />
        <KpiCard title="Dead Stock" value={data.dead_stock.length} accent="coral" />
        <KpiCard title="Overstock" value={data.overstock.length} accent="teal" />
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <ChartCard title="Stockout Risk (<7 days cover)">
          <Table rows={data.stockout_risk} />
        </ChartCard>
        <ChartCard title="Reorder List (<14 days cover)">
          <Table rows={data.reorder_list} />
        </ChartCard>
        <ChartCard title="Dead Stock (no demand)">
          <Table rows={data.dead_stock} />
        </ChartCard>
        <ChartCard title="Overstock (>90 days cover)">
          <Table rows={data.overstock} />
        </ChartCard>
      </div>
    </div>
  )
}
