// A Google-Colab-style strip where pets stroll across a little grassy ground
// line, looping forever. Decorative — hidden from screen readers.
const WALKERS = [
  { pet: "🐕", dur: "13s", delay: "0s", size: "2rem" },
  { pet: "🐈", dur: "17s", delay: "2.5s", size: "1.7rem" },
  { pet: "🐇", dur: "11s", delay: "5s", size: "1.6rem" },
  { pet: "🐢", dur: "22s", delay: "1.2s", size: "1.7rem" },
  { pet: "🦜", dur: "15s", delay: "7s", size: "1.6rem" },
  { pet: "🐕‍🦺", dur: "19s", delay: "9s", size: "2rem" },
  { pet: "🐹", dur: "12s", delay: "11s", size: "1.4rem" },
]

export default function PetParade() {
  return (
    <div
      className="relative w-full h-20 overflow-hidden select-none"
      aria-hidden
    >
      {/* grassy ground line */}
      <div className="absolute bottom-0 left-0 right-0 h-6 bg-leaf/15" />
      <div className="absolute bottom-6 left-0 right-0 border-t-2 border-dashed border-leaf/30" />
      {WALKERS.map((w, i) => (
        <span
          key={i}
          className="absolute bottom-5 animate-walk will-change-transform"
          style={{
            fontSize: w.size,
            animationDuration: w.dur,
            animationDelay: w.delay,
          }}
        >
          {w.pet}
        </span>
      ))}
    </div>
  )
}
