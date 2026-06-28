import { NavLink } from "react-router-dom"
import {
  LayoutDashboard, Boxes, TrendingUp, Truck, Store, BarChart3, Bot, Cog,
} from "lucide-react"

const LINKS = [
  ["/", "Executive", LayoutDashboard],
  ["/inventory", "Inventory", Boxes],
  ["/forecast", "Forecast", TrendingUp],
  ["/suppliers", "Suppliers", Truck],
  ["/stores", "Stores", Store],
  ["/analytics", "Analytics", BarChart3],
  ["/assistant", "AI Assistant", Bot],
  ["/mlops", "MLOps", Cog],
]

export default function Sidebar() {
  return (
    <aside className="w-60 bg-navy text-white min-h-screen p-4 flex flex-col gap-1">
      <div className="text-xl font-bold mb-6 px-2">🐾 Petopia</div>
      {LINKS.map(([to, label, Icon]) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition ${
              isActive ? "bg-teal text-white" : "text-white/70 hover:bg-white/10"
            }`
          }
        >
          <Icon size={18} /> {label}
        </NavLink>
      ))}
      <div className="mt-auto text-[10px] text-white/30 px-2">
        Petopia Intelligence Hub · v1.0
      </div>
    </aside>
  )
}
