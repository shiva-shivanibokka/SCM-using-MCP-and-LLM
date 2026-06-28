import { NavLink } from "react-router-dom"

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

// Two pets stroll back and forth in the slot next to the logo.
const MINI = [
  { pet: "🐕", delay: "0s" },
  { pet: "🐈", delay: "1.6s" },
]

export default function TopBar() {
  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-ink/10">
      <div className="mx-auto max-w-[1840px] px-5 md:px-10">
        <div className="flex items-center gap-5 h-24">
          {/* Logo + strolling pets */}
          <NavLink to="/" className="flex items-center gap-3 shrink-0">
            <div className="relative h-14 w-14 grid place-items-center rounded-2xl bg-teal text-white text-3xl shadow-pop">
              🐾
            </div>
            <div className="leading-tight">
              <div className="font-display text-3xl font-700 text-ink">Petopia</div>
              <div className="text-xs font-bold uppercase tracking-[0.18em] text-teal -mt-0.5">
                Supply Chain IQ
              </div>
            </div>
            <span className="relative ml-2 hidden sm:block w-28 h-8" aria-hidden>
              {MINI.map((m, i) => (
                <span
                  key={i}
                  className="absolute bottom-0 left-0 inline-block animate-walkMini text-2xl"
                  style={{ animationDelay: m.delay }}
                >
                  {m.pet}
                </span>
              ))}
            </span>
          </NavLink>

          {/* Nav */}
          <nav className="flex-1 flex items-center justify-end gap-1.5 overflow-x-auto no-scrollbar">
            {LINKS.map(([to, label, ico]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-2 whitespace-nowrap px-4 py-2.5 rounded-full text-base font-bold transition ${
                    isActive
                      ? "bg-ink text-white shadow-pop"
                      : "text-ink/65 hover:bg-ink/5"
                  }`
                }
              >
                <span className="text-lg" aria-hidden>{ico}</span>
                <span className="hidden lg:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}
