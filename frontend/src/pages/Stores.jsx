import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Stores() {
  const { data } = useQuery({
    queryKey: ["stores"],
    queryFn: () => apiGet("/api/stores/grid"),
  })
  const stores = data?.stores ?? []
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Stores ({stores.length})</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stores.map((s, i) => (
          <ChartCard key={i} title={s.store_id || `Store ${i + 1}`}>
            <div className="text-sm text-navy/70">{s.city || ""}</div>
            <div className="text-xs text-navy/40">
              {s.region || ""} · {s.store_type || ""}
            </div>
            {s.size_sqft && (
              <div className="text-xs text-navy/40 mt-1">{s.size_sqft} sqft</div>
            )}
          </ChartCard>
        ))}
      </div>
    </div>
  )
}
