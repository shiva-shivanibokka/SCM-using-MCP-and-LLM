import { NavLink } from "react-router-dom"

// label, path, emoji — the emoji doubles as the nav icon (quirky + legible).
const LINKS = [
  ["/", "Executive", "📊"],
  ["/inventory", "Inventory", "📦"],
  ["/forecast", "Forecast", "🔮"],
  ["/suppliers", "Suppliers", "🚚"],
  ["/stores", "Stores", "🏬"],
  ["/analytics", "Analytics", "🧁"],
  ["/mlops", "MLOps", "⚙️"],
  ["/assistant", "AI Assistant", "🐕‍🦺"],
]

export default function TopBar() {
  return (
    <header className="sticky top-0 z-30 bg-white/85 backdrop-blur border-b border-ink/10">
      <div className="mx-auto max-w-[1760px] px-5 md:px-10">
        <div className="flex items-center gap-4 h-16">
          {/* Logo + trotting mascot */}
          <NavLink to="/" className="flex items-center gap-2 shrink-0">
            <div className="relative h-9 w-9 grid place-items-center rounded-2xl bg-teal text-white text-xl shadow-pop">
              🐾
            </div>
            <div className="leading-tight">
              <div className="font-display text-xl font-700 text-ink">Petopia</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-teal -mt-1">
                Supply Chain IQ
              </div>
            </div>
            <span
              className="ml-1 hidden sm:inline-block w-32 overflow-hidden"
              aria-hidden
            >
              <span className="inline-block animate-trot text-lg">🐕</span>
            </span>
          </NavLink>

          {/* Nav */}
          <nav className="flex-1 flex items-center gap-1 overflow-x-auto no-scrollbar">
            {LINKS.map(([to, label, ico]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 whitespace-nowrap px-3 py-2 rounded-full text-sm font-bold transition ${
                    isActive
                      ? "bg-ink text-white shadow-pop"
                      : "text-ink/65 hover:bg-ink/5"
                  }`
                }
              >
                <span aria-hidden>{ico}</span>
                <span className="hidden md:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}
