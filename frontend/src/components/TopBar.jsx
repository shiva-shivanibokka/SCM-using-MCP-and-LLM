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

// Pets stroll along the header's bottom edge, near the logo — absolutely
// positioned so they never steal width from the nav.
const STROLLERS = [
  { pet: "🐕", left: "0px", delay: "0s" },
  { pet: "🐈", left: "70px", delay: "1.1s" },
  { pet: "🐇", left: "140px", delay: "2.2s" },
]

export default function TopBar() {
  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-ink/10 relative">
      <div className="mx-auto max-w-[1840px] px-5 md:px-10">
        <div className="flex items-center gap-6 h-24">
          {/* Logo */}
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
          </NavLink>

          {/* Nav */}
          <nav className="flex-1 flex items-center justify-end gap-1 overflow-x-auto no-scrollbar">
            {LINKS.map(([to, label, ico]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-2 whitespace-nowrap px-3.5 py-2.5 rounded-full text-base font-bold transition ${
                    isActive
                      ? "bg-ink text-white shadow-pop"
                      : "text-ink/65 hover:bg-ink/5"
                  }`
                }
              >
                <span className="text-lg" aria-hidden>{ico}</span>
                <span className="hidden xl:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      {/* Strolling pets along the bottom edge near the logo */}
      <div
        className="pointer-events-none absolute bottom-0 left-5 md:left-10 w-52 h-7 overflow-hidden hidden sm:block"
        aria-hidden
      >
        {STROLLERS.map((s, i) => (
          <span
            key={i}
            className="absolute bottom-0 inline-block animate-walkMini text-xl"
            style={{ left: s.left, animationDelay: s.delay }}
          >
            {s.pet}
          </span>
        ))}
      </div>
    </header>
  )
}
