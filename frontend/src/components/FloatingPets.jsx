// Ambient layer: a few pet emojis drift slowly upward behind the content.
// Purely decorative — pointer-events-none and hidden from screen readers.
const PETS = ["🐶", "🐱", "🐰", "🐹", "🐦", "🐠", "🐢", "🦴", "🐾"]

export default function FloatingPets({ count = 5 }) {
  const items = Array.from({ length: count }, (_, i) => ({
    pet: PETS[i % PETS.length],
    left: `${(i * 13 + 6) % 96}%`,
    delay: `${(i * 1.7) % 9}s`,
    duration: `${8 + (i % 5)}s`,
    size: `${1.1 + (i % 3) * 0.4}rem`,
  }))
  return (
    <div
      className="pointer-events-none fixed inset-0 overflow-hidden -z-10"
      aria-hidden
    >
      {items.map((it, i) => (
        <span
          key={i}
          className="absolute bottom-0 animate-floatUp opacity-0 select-none"
          style={{
            left: it.left,
            fontSize: it.size,
            animationDelay: it.delay,
            animationDuration: it.duration,
          }}
        >
          {it.pet}
        </span>
      ))}
    </div>
  )
}
